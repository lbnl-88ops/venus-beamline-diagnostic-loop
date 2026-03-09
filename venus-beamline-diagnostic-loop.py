#!/bin/env python3
# coding=ASCII
import asyncio

import inspect
import itertools
import time
import types
import signal
import numpy as np
import datetime
import time
import statistics
from labjack import ljm  # for communication with LabJack
import os

from dotenv import dotenv_values
import pyvisa
from pyvisa.resources.messagebased import MessageBasedResource as Connection

from ops.ecris.devices.motor_controller import MotorController, Axis
from ops.ecris.drivers.keithley import Keithley, SCPIDriver
import venus_data_utils.venusplc as venusplc

env_config = dotenv_values(".env")
venus = venusplc.VENUSController(read_only=False)

# default M/Q range used for fast CSD button on main tuning screen
mq_min_default = 0.84
mq_max_default = 8.90
n_csd_steps_default = 1200

# set directory to save csds
directory = "/data/csds/"
emittance_scan_directory = "/data/emittance/"
MODULE_1_USB = "USB0::1510::29970::04684146\x00\x00::0::INSTR"
MODULE_2_USB = "USB0::1510::29970::04684147\x00\x00::0::INSTR"

# names that are strings
stringnames = ["beam_element"]

############## stuff to set up faster Ammeter
measurementFrequency = 1000


def sendCommand(connection: Connection, command):
    connection.write(command)


def setupSystem(verbose=0):
    # Connect to Ammeter
    if verbose:
        print("attempt to connect...")
    rm = pyvisa.ResourceManager()
    connection: Connection = rm.open_resource(MODULE_2_USB)
    # sendCommand(connection,'*lang scpi')
    # output = connection.query('*idn?')
    # if verbose:   print('connected.  Output: ',output,'\nResetting system')

    # Reset System
    sendCommand(connection, "*rst")
    if verbose:
        print("reset")
    time.sleep(2)
    if verbose:
        print("waited 2 seconds, setting up current reading")

    # Setting up reading settings
    sendCommand(connection, ':sens:func "curr"')
    # sendCommand(connection,':sens:curr:rang:auto on')
    sendCommand(connection, ":sens:curr:rang 1e-3")
    sendCommand(connection, ":sens:curr:azer off")
    sendCommand(connection, ":sens:curr:nplc:auto off")

    # Set integration time in terms of wall frequency: MeasTime*^60Hz
    nplc = 1.0 / measurementFrequency * 60.0
    sendCommand(connection, ":sens:curr:nplc " + str(nplc))
    # sendCommand(connection, ':sens:curr:aper: 1e-3')

    # turn on input switch
    # sendCommand(connection,':inp on')

    return connection


async def connect_motor_controller() -> MotorController:
    try:
        temp_port = env_config["MOTOR_CONTROLLER_PORT"]
        if temp_port is None:
            raise KeyError(".env file requires MOTOR_CONTROLLER_PORT")
        port = int(temp_port)
    except ValueError:
        raise ValueError("Invalid MOTOR_CONTROLLER_PORT")
    ip = env_config["MOTOR_CONTROLLER_PORT"]
    if ip is None:
        raise KeyError(".env file requires MOTOR_CONTROLLER_PORT")
    motor_controller = MotorController(ip=ip, port=port)
    await motor_controller.connect()
    return motor_controller


async def connect_keithley(module_usb: str) -> Keithley:
    keithley = Keithley.connect_at_usb(resource_name=module_usb, aperture_time=1e-3)
    await keithley.connect()
    await keithley.send_silent_command(SCPIDriver.Commands.VOLTAGE_AUTOZERO_OFF)
    await keithley.send_silent_command(SCPIDriver.Commands.VOLTAGE_DELAY_DISABLE)
    await keithley.send_silent_command(SCPIDriver.Commands.VOLTAGE_AUTO_RANGE_OFF)
    await keithley.send_silent_command(SCPIDriver.Commands.set_voltage_range(100e-3))
    return keithley


def getCurrent(connection: Connection):
    return float(connection.query(":meas:curr?"))
    # sendCommand(connection,":meas:curr?")
    # return float(connection.read_until(b'\n').decode("ascii")[-15:-2])


connection = setupSystem(verbose=0)
motor_controller = asyncio.run(connect_motor_controller())
emittance_keithley = asyncio.run(connect_keithley(MODULE_1_USB))

################ done setting up faster Ammeter
################ stuff to set up LabJack

handle = ljm.openS("T8", "usb", "ANY")


def getB():
    B = ljm.eReadName(handle, "AIN0")
    return B * 0.4  # hall probe has 2 T as 5 V


def setBatman(current):
    ljm.eWriteName(handle, "DAC0", current * 0.04)


################  done setting up Labjack

################  set up stuff for CSDs

dipolealpha = 0.00823

if not os.path.exists("csds"):
    os.makedirs("csds")
if not os.path.exists(directory):
    os.makedirs(directory)


def datasheet(tst_str):
    readvars = venus.read_vars()
    #    for i in range(len(readvars)):
    #        print(f'{i:3d} {readvars[i]} {venus.read([readvars[i]])}')
    with open(directory + "/dsht_" + tst_str, "w") as f:
        for i in range(len(readvars)):
            if i != 8:
                f.write("%4i %.5e %s\n" % (i, venus.read([readvars[i]]), readvars[i]))


def get_csd(Ilow, Ihigh, npoints, wasin, tstarted):
    sendCommand(connection, ":sens:azer:once")
    Vext = venus.read(["extraction_v"])
    Bstart = getB()
    Istart = venus.read(["batman_i_set"]) / 131.0

    changeslow(Istart, Ilow, twait=1)
    if wasin == 0:
        morewaittime = tstarted + 10.0 - time.time()
        if morewaittime > 0:
            if morewaittime > 10:  # add this as a safety catch.  Shouldn't happen
                print(f"morewaittime > 10!! {morewaittime:.2f}")
                time.sleep(10)
            else:
                time.sleep(morewaittime)

    batmanfield = np.zeros(npoints)
    faradaycup = np.zeros(npoints)
    timesteps = np.zeros(npoints)

    ipoints = np.sqrt(np.linspace(Ilow * Ilow, Ihigh * Ihigh, npoints))
    for i in range(npoints):
        setBatman(ipoints[i])
        faradaycup[i] = getCurrent(connection)
        batmanfield[i] = getB()
        timesteps[i] = time.time()

    if wasin == 0:
        venus.write(
            {"fcv1_in": False}
        )  # Take Faraday cup back out if it was out before start
    changeslow(ipoints[-1], Istart, twait=0)
    resetbatman(Bstart, Istart)

    # add search to peak beam here
    return (timesteps, ipoints, batmanfield, faradaycup)


def resetbatman(bgoal, Istart):
    tstart = time.time()

    # with open('temptemp','w') as f:
    if 1:
        Inew = Istart
        while time.time() < tstart + 7.5:
            bnow = getB()
            if bnow < bgoal:
                Inew = Inew + 0.007
            elif bnow > bgoal:
                Inew = Inew - 0.007
            # f.write(f"{time.time()-tstart:7.4f} {Inew:10.3f} {bnow*1e5:10.3f} {1e6*getCurrent(connection):10.3f}\n")
            setBatman(Inew)
    if 0:  # Use this to make final steps if determined necessary
        Inew = Inew + 0.0
        setBatman(Inew)


def changeslow(istart, iend, twait=1):
    ipts = np.linspace(istart, iend, int(np.ceil(np.abs(istart - iend))) * 3)
    for ipt in ipts:
        setBatman(ipt)
        iNow = getCurrent(connection)  # doing this to slow the process
        Bnow = getB()  # doing this to slow the process
    time.sleep(twait)


def quickave(num=30):
    tot = 0
    for i in range(num):
        tot = tot + getCurrent(connection)
    return tot / (num * 1.0)


def performFastCSD(mqmin, mqmax, npts_csd):
    # take a datasheet and a csd
    tallstart = time.time()
    tnowstr = str(int(time.time()))
    wasin = 1
    if not venus.read(["fcv1_in"]):  # checking if faraday cup is in
        venus.write({"fcv1_in": True})  # if not, put it in
        wasin = 0  # set wasin to zero.  Will have to wait ~10 s to start CSD
    datasheet(tnowstr)

    alpha = 0.00824  # calculated...need notes DST
    m = 79e-5  # slope of linear fit of B vs I...need notes DST
    Vext = venus.read({"extraction_v"})

    Ilow = alpha / m * np.sqrt(mqmin * Vext)
    Ihigh = alpha / m * np.sqrt(mqmax * Vext)
    nsteps = npts_csd
    if Ilow > 250.0:
        Ilow = 240.0
    if Ihigh > 250.0:
        Ihigh = 250.0

    with open(directory + "/csd_" + tnowstr, "w") as outfile:
        timesteps, ipoints, batmanfield, faradaycup = get_csd(
            Ilow, Ihigh, nsteps, wasin, tallstart
        )
        for i in range(len(timesteps)):
            outfile.write(
                "%.3f %.3f %.8f %.5e\n"
                % (timesteps[i], ipoints[i], batmanfield[i], faradaycup[i])
            )
    tnow = time.time()
    nowdt = datetime.datetime.now()
    formatted_time = nowdt.strftime("%Y-%m-%d %H:%M:%S")
    with open(directory + "log", "a") as f:
        f.write(f"{formatted_time} CSD time = {time.time() - tallstart:.1f}\n")
    print(f"{formatted_time} CSD time = {time.time() - tallstart:.1f}")


def maximizeCurrent():
    # parameters to set
    dt_collect = 0.25  # how long to average beam current
    dt_wait = 0.3  # how long to wait after changing Batman current
    fractionofmax = [0.9, 0.95]  # fraction of maximum to know you are past one side
    dbatman = 0.007  # step size for batman
    stepsize = 8  # number of dbatman steps to make in initial search for peak

    twhentostop = time.time() + 40.0  # stop if this takes more than 40 seconds

    steplast = 0
    for j in range(2):
        magnetsteps = []
        meancurrent = []
        maxcurrent = 0.0
        ibatmanstart = venus.read(["batman_i_set"]) / 131.0

        newmean = quickave()
        meancurrent.append(newmean)
        magnetsteps.append(0)
        maxcurrent = newmean

        if j == 1:
            stepsize = 1
            steplast = 0
        for i in range(2):
            dstep = stepsize * (-1) ** i
            while newmean >= fractionofmax[j] * maxcurrent:
                steplast = steplast + dstep
                setBatman(
                    ibatmanstart + steplast * dbatman
                )  # VENUS minimum step sizes are 0.007 A
                time.sleep(dt_wait)
                # newmean, newstd = getmeancurrent(dt_collect)
                newmean = quickave()
                meancurrent.append(newmean)
                magnetsteps.append(steplast)
                maxcurrent = max(meancurrent)
                if time.time() > twhentostop:
                    break

            measuredmax = max(meancurrent)
            maxindex = meancurrent.index(measuredmax)
            steplast = magnetsteps[maxindex]
            setBatman(ibatmanstart + steplast * dbatman)
            venus.write({"batman_i": ibatmanstart + steplast * dbatman})
            newmean = (
                maxcurrent  # artificially set this as it should be close to correct
            )


# reset just in case
venus.write({"csd_in_progress": 0})

again = 1
ibatmanlast = venus.read(["batman_i"])
treadagain = time.time()
tlastave = time.time()
tlastautozero = time.time()
last_scaling_factor = None
while again:
    if time.time() - tlastautozero > 600:
        sendCommand(connection, ":sens:azer:once")
        tlastautozero = time.time()
        tlastave = time.time()
    nmeas = 0.0
    iave = 0.0
    isq = 0.0
    while time.time() - tlastave < 0.33:
        nmeas = nmeas + 1
        inow = getCurrent(connection)
        iave = iave + inow
        isq = isq + inow * inow

        # check for batman requests
        ibatmanrequest = venus.read(["batman_i_set"]) / 131.0
        if ibatmanrequest != ibatmanlast:
            if ibatmanrequest > (ibatmanlast + 1) or ibatmanrequest < (ibatmanlast - 1):
                changeslow(ibatmanlast, ibatmanrequest, twait=0)
            else:
                setBatman(ibatmanrequest)
            ibatmanlast = ibatmanrequest
    tlastave = time.time()
    iave = iave / (nmeas)
    isq = isq / (nmeas)
    istd = np.sqrt(isq - iave * iave)
    venus.write({"fcv1_ammeter": iave})
    if iave == 0:
        venus.write({"fcv1_ammeter_stdev": -2.0})
    else:
        venus.write({"fcv1_ammeter_stdev": istd / iave * 100.0})

    request_csd = 0
    if venus.read(["csd_request"]):
        request_csd = 1
    if venus.read(["csd_custom_request"]):
        request_csd = 2

    if request_csd:
        venus.write(
            {"fcv1_ammeter_stdev": 0.0}
        )  # set to zero as an indicator csd is happening

        if request_csd == 1:
            venus.write({"csd_in_progress": 1})
            performFastCSD(mq_min_default, mq_max_default, n_csd_steps_default)
            venus.write({"csd_in_progress": 0})
        if request_csd == 2:
            venus.write({"csd_custom_in_progress": 1})
            performFastCSD(
                venus.read(["csd_MQ_min"]),
                venus.read(["csd_MQ_max"]),
                venus.read(["num_csd_points"]),
            )
            venus.write({"csd_custom_in_progress": 0})

        tlastave = time.time()
        ibatmanrequest = venus.read(["batman_i_set"]) / 131.0  # new V3
        ibatmanlast = ibatmanrequest  # new: setting equal after return so code doesn't make change after peaking

    if venus.read(["peaking_request"]):
        venus.write({"peaking_in_progress": 1})
        ibatman_pre_request = venus.read(["batman_i_set"]) / 131.0  # new V3
        bbatman_pre_request = getB()
        tpeaking = time.time()
        maximizeCurrent()
        tlastave = time.time()
        ibatmanrequest = venus.read(["batman_i_set"]) / 131.0  # new V3
        bbatmanrequest = getB()
        ibatmanlast = ibatmanrequest
        venus.write({"peaking_in_progress": 0})
        ##  Eventually get rid of the next lines.  Just using this to diagnose how peaking is working
        with open("peaking_results", "a") as fff:
            fff.write(
                "%i %6.3f %7.3f %.5f %7.3f %.5f\n"
                % (
                    int(tpeaking),
                    tlastave - tpeaking,
                    ibatman_pre_request,
                    bbatman_pre_request,
                    ibatmanrequest,
                    bbatmanrequest,
                )
            )

    if venus.read(["emittance_retract_axis"]):
        asyncio.run(motor_controller.move_axis_to_positive_eof(Axis.VenusX))
        asyncio.run(motor_controller.move_axis_to_positive_eof(Axis.VenusY))

    if venus.read(["emittance_scan_request"]):
        faraday_cup_in = 0
        if venus.read(["fcv1_in"]):  # checking if faraday cup is in
            venus.write({"fcv1_in": False})  # if it is, take it out
            faraday_cup_in = 1
            time.sleep(1)  # let it start going out
            while venus.read(["fcv1_in"]):
                time.sleep(0.1)

        tscanstart = time.time()
        venus.write({"emittance_scan_in_progress": 1})
        # This function should be fiddled with by Jessica.  It is being passed, in
        #   order: direction, position min, position max, position step size,
        #          divergence min, divergence max, divergence step size, and
        #          the keithley multiplier integer (e.g. 2 for 1E2), and a flat to leave the scanner in
        # Note that some care will have to be taken with the min/max/step groupings if something
        #    like the following is requested: -10,10,3.  What I would suggest is to round up.
        #    For example, with -10,10,3, there are 7.666 steps, and what I would do is round up
        #    so that this is really np.linspace(-10,10,int(ceiling((max-min)/stepsize)+1))
        leave_scanner_in = venus.read(["emittance_leave_in"])
        JessicaCallsMagicEmittance(
            venus.read(["emittance_direction"]),
            venus.read(["emittance_position_min"]),
            venus.read(["emittance_position_max"]),
            venus.read(["emittance_position_step"]),
            venus.read(["emittance_divergence_min"]),
            venus.read(["emittance_divergence_max"]),
            venus.read(["emittance_divergence_step"]),
            venus.read(["emittance_keithley_multiplier"]),
            leave_scanner_in,
        )
        ### NOTE: keithley multiplier is an integer:  turn to 10Einteger
        venus.write({"emittance_scan_in_progress": 0})
        formatted_time = nowdt.strftime("%Y-%m-%d %H:%M:%S")
        with open(directory + "log", "a") as f:
            f.write(
                f"{formatted_time} emittance scan time = {time.time() - tscanstart:.1f}\n"
            )
        print(f"{formatted_time} emittance scan time = {time.time() - tscanstart:.1f}")
        if faraday_cup_in == 1 and not (leave_scanner_in):
            venus.write({"fcv1_in": True})  # put Faraday Cup back in

    if time.time() - treadagain >= 5:
        with open("again", "r") as f:
            again = int(f.readline())
        treadagain = time.time()
        tlastave = time.time()

###  done with CSD functions
connection.close()  # close connnection to Ammeter
ljm.close(handle)  # close connection to labjack
