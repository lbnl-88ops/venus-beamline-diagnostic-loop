# Agent Guide: Venus Beamline Diagnostic Loop

## High-Signal Facts

### Execution & Entry Point
- **Main Entry Point:** `venus-beamline-diagnostic-loop.py`
- **Primary Command:** `poetry run python venus-beamline-diagnostic-loop.py`
- **Virtual Environment:** A local virtual environment is available at `.venv/bin/activate`.
- **Known Issue:** The script contains a relative import `from .src.emittance_scan import EmittanceScanManager`. This causes an `ImportError` when run directly. You may need to remove the leading dot or run from the parent directory as a package (though the filename has dashes).

### Graceful Shutdown
- This script uses a manual file-based switch for graceful termination.
- To stop the loop cleanly: Edit the file named `again` in the root directory and change its content from `1` to `0`. The script checks this file every 5 seconds.

### Environment & Dependencies
- **Environment Variables:** Requires a `.env` file (see `venus-beamline-diagnostic-loop.py` for variables like `MOTOR_CONTROLLER_IP`, `MOTOR_CONTROLLER_PORT`).
- **Ethernet Fallback:** Add `MODULE_1_IP` and `MODULE_2_IP` to `.env` for Keithley ammeter ethernet fallback support.
- **Hardware Drivers:** Requires LabJack LJM drivers (`libLabJackM.so`). Execution will fail in environments without these drivers.
- **Internal Libraries:** Depends on `ops-ecris`, installed via GitHub in `pyproject.toml`.

### Resilience Features
- **Timeouts:** VISA resources have a default timeout of 5000ms to detect USB stalls.
- **Ammeter Ethernet Fallback:** The Keithley ammeters will automatically fall back to Ethernet if the USB connection fails.
- **USB Recovery Check:** Periodic checks (configurable via `USB_RECONNECT_CHECK_INTERVAL` in `.env`, defaults to 60s) attempt to switch back to USB if the device has recovered.
- **Targeted Error Handling:** Reconnection logic is specifically applied to the Ammeter connections to address observed USB instability without adding broad complexity to other hardware paths.

### Architecture & Data
- **Core Logic:** The main script orchestrates hardware (Keithley ammeter, LabJack, Venus PLC, Motor Controller).
- **Scan Management:** Complex emittance scans are managed by `src/emittance_scan.py`.
- **Output:** Data is stored in the `csds/` directory (Charge State Distributions).

## Workflow Quirks
- **No Tests/Linting:** There are no configured test suites or linting tools in `pyproject.toml`.
- **"Cassette Futurism" Design:** The code is functional and tactile (e.g., the `again` file switch). Maintain this directness when making changes.
