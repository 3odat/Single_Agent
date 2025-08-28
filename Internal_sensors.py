#!/usr/bin/env python3
"""
sensors_service.py
-------------------

This script provides a small telemetry service that reads internal sensor data from
a PX4 flight stack using MAVSDK, prints a human‑readable summary to the
terminal every second, writes the latest telemetry snapshot to disk as JSON,
and exposes a FastAPI endpoint (`/sensors`) so that other programs (for
example, a LangGraph agent) can query the most recent snapshot in real time.

The code is derived from `internal_sensors.py` and reuses its Monitor class
for sensor subscriptions and connection management.  The key differences are
that the built‑in dashboard ticker has been replaced with a custom printer
that formats the output according to a simple example, and the monitor and
printer run alongside a FastAPI web server using asyncio.

Usage:
    python sensors_service.py --url udp://:14540 --hz 1.0 --json mavsdk_sensor_snapshot.json

You can then point your browser or agent at http://localhost:8001/sensors to
retrieve the latest telemetry snapshot as JSON.  The terminal will update
once per second with a numbered reading and timestamp, and the JSON file will
be rewritten on the same schedule.
"""

import argparse
import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI

try:
    from internal_sensors import Monitor, now_iso
except ImportError:
    # Fall back to local definitions if internal_sensors isn't available.
    from internal_sensors import Monitor, now_iso  # type: ignore


def fmt(value: Optional[Any], nd: int = 2, unit: str = "", percent: bool = False) -> str:
    """Helper to format numbers for display.  If value is None, returns "N/A"."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if percent:
        return f"{v:.{nd}f}%"
    return f"{v:.{nd}f}{unit}"


class CustomMonitor(Monitor):
    """
    A wrapper around the original Monitor that replaces the dashboard ticker
    with a custom printer producing output matching the requested format.  It
    also writes the snapshot to a JSON file each time it prints and keeps a
    counter of how many readings have been printed.
    """

    def __init__(self, url: str, hz: float, json_path: Path) -> None:
        super().__init__(url=url, hz=hz, json_path=json_path)
        self._count: int = 0

    def _render_custom(self) -> str:
        """Build a human‑readable summary of the current snapshot.

        The output follows the example provided in the user request.  It
        includes a numbered reading, a time stamp with minutes and seconds,
        battery status, GPS info, position, velocity, attitude, flight status,
        system health, remote control status, and navigation data.  Missing
        values are represented by "N/A" or "Unknown" as appropriate.
        """
        snap = self.snap
        # Compute a friendly time string (HH:MM.SS) from the timestamp.  Use
        # the current time if the snapshot has not yet been populated.
        try:
            ts_str = snap.get("timestamp") or now_iso()
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
            time_str = ts_dt.strftime("%H:%M.%S")
        except Exception:
            time_str = datetime.now(timezone.utc).strftime("%H:%M.%S")

        # Battery
        batt = snap.get("battery", {})
        batt_rem = batt.get("remaining")
        battery_charge = fmt(batt_rem * 100.0 if isinstance(batt_rem, (int, float)) else None, nd=1, unit="%")
        battery_voltage = fmt(batt.get("voltage_v"), nd=2, unit=" V")
        battery_current = fmt(batt.get("current_a"), nd=2, unit=" A")

        # GPS
        gps = snap.get("gps", {})
        gps_fix = gps.get("fix_type", "N/A") or "N/A"
        gps_sats = gps.get("num_satellites")
        gps_sats_str = str(gps_sats) if gps_sats is not None else "N/A"
        gps_signal = "Unknown"  # no quality metric available in this snapshot

        # Position
        pos = snap.get("position", {})
        pos_lat = fmt(pos.get("lat_deg"), nd=6)
        pos_lon = fmt(pos.get("lon_deg"), nd=6)
        pos_alt = fmt(pos.get("rel_alt_m"), nd=2, unit=" m")

        # Velocity
        vel = snap.get("velocity_ned", {})
        vel_n = fmt(vel.get("north_m_s"), nd=2, unit=" m/s")
        vel_e = fmt(vel.get("east_m_s"), nd=2, unit=" m/s")
        vel_d = fmt(vel.get("down_m_s"), nd=2, unit=" m/s")

        # Attitude
        att = snap.get("attitude", {}).get("euler_deg", {})
        att_roll = fmt(att.get("roll_deg"), nd=1, unit="°")
        att_pitch = fmt(att.get("pitch_deg"), nd=1, unit="°")
        att_yaw = fmt(att.get("yaw_deg"), nd=1, unit="°")

        # Flight status
        stat = snap.get("status", {})
        stat_mode = stat.get("flight_mode", "N/A")
        stat_armed = str(stat.get("armed", "N/A"))
        stat_in_air = str(stat.get("in_air", "N/A"))

        # System health
        hlth = snap.get("health", {})
        def health_str(key: str) -> str:
            val = hlth.get(key)
            if val is None:
                return "N/A"
            return "OK" if bool(val) else "Not OK"
        health_local = health_str("local_position_ok")
        health_global = health_str("global_position_ok")
        health_home = health_str("home_position_ok")

        # Remote control
        rc = snap.get("rc", {})
        rc_avail = str(rc.get("available", "N/A"))
        rc_strength = fmt(rc.get("signal_strength_percent"), nd=0, unit="%")

        # Navigation
        nav_heading = fmt(snap.get("heading_deg"), nd=1, unit="°")
        wind = snap.get("wind", {})
        nav_wind_speed = fmt(wind.get("speed_m_s"), nd=1, unit=" m/s")
        nav_wind_dir = fmt(wind.get("direction_deg"), nd=0, unit="°")

        # Build multi‑line output
        lines = []
        # Reading counter with zero‑pad to three digits
        count_str = f"{self._count:03d}" if self._count < 1000 else str(self._count)
        lines.append(f"Sensor Reading {count_str}:")
        lines.append(f"Time: {time_str}\n")
        lines.append("Battery Status:")
        lines.append(f"Charge Level: {battery_charge}")
        lines.append(f"Voltage: {battery_voltage}")
        lines.append(f"Current: {battery_current}\n")
        lines.append("GPS Information:")
        lines.append(f"Fix Status: {gps_fix}")
        lines.append(f"Satellites: {gps_sats_str}")
        lines.append(f"Signal Quality: {gps_signal}\n")
        lines.append("Position Data:")
        lines.append(f"Latitude: {pos_lat}")
        lines.append(f"Longitude: {pos_lon}")
        lines.append(f"Altitude (Relative): {pos_alt}\n")
        lines.append("Velocity Vectors:")
        lines.append(f"North: {vel_n}")
        lines.append(f"East: {vel_e}")
        lines.append(f"Down: {vel_d}\n")
        lines.append("Attitude Information:")
        lines.append(f"Roll: {att_roll}")
        lines.append(f"Pitch: {att_pitch}")
        lines.append(f"Yaw: {att_yaw}\n")
        lines.append("Flight Status:")
        lines.append(f"Mode: {stat_mode}")
        lines.append(f"Armed: {stat_armed}")
        lines.append(f"In Air: {stat_in_air}\n")
        lines.append("System Health:")
        lines.append(f"Local Position: {health_local}")
        lines.append(f"Global Position: {health_global}")
        lines.append(f"Home Position: {health_home}\n")
        lines.append("Remote Control:")
        lines.append(f"Available: {rc_avail}")
        lines.append(f"Signal Strength: {rc_strength}\n")
        lines.append("Navigation:")
        lines.append(f"Heading: {nav_heading}")
        lines.append(f"Wind Speed: {nav_wind_speed}")
        lines.append(f"Wind Direction: {nav_wind_dir}")
        return "\n".join(lines)

    async def _printer(self) -> None:
        """Periodically update timestamp, write JSON and print custom telemetry."""
        # Use a 1 Hz interval regardless of self.hz; the user requested real‑time
        # updates every second.  If you need a different rate, change the
        # interval value here.
        interval = 1.0
        while not self._stop.is_set():
            # Update the timestamp in the snapshot to the current time
            self.snap["timestamp"] = now_iso()
            # Increment reading counter
            self._count += 1
            # Serialize snapshot to JSON and write to disk
            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    # Replace NaN or infinite values with None to avoid JSON errors
                    safe = json.loads(json.dumps(self.snap, default=lambda o: None))
                    json.dump(safe, f, indent=2)
            except Exception as exc:
                print(f"[WARN] Failed to write JSON snapshot: {exc}")
            # Print custom summary to the terminal
            print("\x1b[2J\x1b[H", end="")  # clear screen and move cursor home
            print(self._render_custom(), flush=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def run_custom(self) -> None:
        """Run the sensor monitor, custom printer, and handle shutdown."""
        await self.connect()
        # Start individual sensor subscriber tasks
        tasks = [
            asyncio.create_task(self._battery()),
            asyncio.create_task(self._gps()),
            asyncio.create_task(self._position()),
            asyncio.create_task(self._velocity()),
            asyncio.create_task(self._attitude()),
            asyncio.create_task(self._health()),
            asyncio.create_task(self._status()),
            asyncio.create_task(self._rc()),
            asyncio.create_task(self._heading()),
            asyncio.create_task(self._distance_sensor()),
            asyncio.create_task(self._wind()),
            asyncio.create_task(self._status_text()),
            asyncio.create_task(self._odometry()),
            asyncio.create_task(self._printer()),
        ]
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass
        # Wait until stop event is triggered
        await self._stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("[INFO] Monitor stopped.")


def create_app(monitor: CustomMonitor) -> FastAPI:
    """
    Create a FastAPI application that exposes the current sensor snapshot.

    The API has a single endpoint `/sensors` which returns the entire
    snapshot dictionary.  Additional endpoints could easily be added to
    expose individual sensors if required.
    """
    app = FastAPI()

    @app.get("/sensors")
    async def get_sensors() -> Dict[str, Any]:  # pragma: no cover
        # Return a deep copy of the snapshot to avoid concurrent modification
        try:
            # Use json.loads(json.dumps(...)) to ensure NaN and other non‑JSON
            # values are coerced to None
            return json.loads(json.dumps(monitor.snap, default=lambda o: None))
        except Exception as exc:
            return {"error": str(exc)}

    return app


async def main_async(url: str, hz: float, json_path: Path, host: str, port: int) -> None:
    """Entry point to launch the custom monitor and API concurrently."""
    monitor = CustomMonitor(url=url, hz=hz, json_path=json_path)
    app = create_app(monitor)

    # Start the monitor in a background task
    monitor_task = asyncio.create_task(monitor.run_custom())

    # Run the FastAPI server.  Use uvicorn's programmatic API so it runs
    # inside the existing asyncio event loop.
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info", lifespan="on")
    server = uvicorn.Server(config=config)
    server_task = asyncio.create_task(server.serve())

    # Wait for either task to finish (monitor stops or server stops)
    done, pending = await asyncio.wait(
        [monitor_task, server_task], return_when=asyncio.FIRST_COMPLETED
    )
    # Cancel the remaining tasks if one completes
    for task in pending:
        task.cancel()
    # Wait for cancellation to propagate
    await asyncio.gather(*pending, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PX4 telemetry service with custom output and API")
    parser.add_argument(
        "--url", default="udp://:14540", help="MAVSDK connection URL (e.g., udp://:14540 or serial:///dev/ttyACM0:115200)"
    )
    parser.add_argument(
        "--hz", type=float, default=1.0, help="Refresh rate in Hz for updating the snapshot and printing output"
    )
    parser.add_argument(
        "--json", type=Path, default=Path("mavsdk_sensor_snapshot.json"), help="Path to write the latest snapshot JSON"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the FastAPI server"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to bind the FastAPI server"
    )
    args = parser.parse_args()
    try:
        asyncio.run(main_async(url=args.url, hz=args.hz, json_path=args.json, host=args.host, port=args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
