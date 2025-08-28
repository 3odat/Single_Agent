#!/usr/bin/env python3
"""
Robust MAVSDK drone controller with interactive CLI and improved offboard
behaviour.  This script expands on earlier versions by unifying colour
handling, preventing drift after maneuvers via a short zero‑velocity
settling period, and providing more informative console output.

Key Features
------------

* **Automatic connection & arming** – on startup the controller
  connects to the default UDP endpoint (``udp://:14540``), waits
  until a global position fix and home position are established,
  arms the drone, and prints progress.
* **Comprehensive motion primitives** – move in the X/Y/Z body axes,
  rotate clockwise/counter‑clockwise, fly circles, squares, hover,
  fly to GPS coordinates, orbit around a point, and more.  All
  velocity‑based maneuvers use a continuous stream of setpoints and
  include a settling phase to avoid residual drift【392627369154405†L754-L804】.
* **Mission utilities** – record and replay flight paths as mission
  waypoints, set geofence boundaries along a road, patrol back and
  forth, and perform a return‑to‑launch (RTL).
* **Rich feedback** – coloured status messages indicate success,
  warnings or errors, and snapshot telemetry can be printed on demand.

This controller is designed to be both a standalone CLI and a
reusable library for agent‑based architectures (e.g. LangGraph or
LangChain).  The code follows the MAVSDK Python guidelines for
offboard control, where velocity commands are sent continuously at
around 20 Hz during manoeuvres and a zero‑velocity stream is used
afterwards to ensure the vehicle holds position【392627369154405†L754-L804】.
"""

import asyncio
import math
import time
from datetime import datetime
from typing import List, Optional

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from mavsdk.action import ActionError
from mavsdk.mission import MissionItem, MissionPlan
from mavsdk.telemetry import LandedState
from mavsdk.geofence import Point


class Colors:
    """ANSI colour codes for terminal output.

    Both ``END`` and ``ENDC`` are provided as aliases to reset
    formatting.  ``BOLD`` can be combined with the colours to
    emphasise messages.
    """

    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'
    ENDC = '\033[0m'


class DroneAgentController:
    """High level controller exposing many skills for a PX4/MAVSDK drone."""

    def __init__(self) -> None:
        self.drone: System = System()
        self.offboard_active: bool = False
        self.armed: bool = False
        self.home_position = None
        self.recorded_positions: List[dict] = []
        self.is_recording: bool = False
        self.log_file = f"drone_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    async def connect(self, address: str = "udp://:14540") -> None:
        """Connect to the drone and wait for a global position fix.

        Parameters
        ----------
        address: str, optional
            MAVLink address to connect to. Defaults to ``udp://:14540``.
        """
        print(f"{Colors.BLUE}🔌 Connecting to drone at {address}…{Colors.ENDC}")
        await self.drone.connect(system_address=address)
        print(f"{Colors.YELLOW}⏳ Waiting for connection…{Colors.ENDC}")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"{Colors.GREEN}✅ Connected to drone!{Colors.ENDC}")
                break
            await asyncio.sleep(0.1)
        print(f"{Colors.YELLOW}⏳ Waiting for global position and home fix…{Colors.ENDC}")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print(f"{Colors.GREEN}✅ Global position OK{Colors.ENDC}")
                break
            await asyncio.sleep(0.5)
        # Store home position
        async for home in self.drone.telemetry.home():
            self.home_position = home
            break
        print(f"{Colors.GREEN}🏠 Home position set.{Colors.ENDC}")

    async def arm(self) -> None:
        """Arm the drone with retries."""
        if self.armed:
            print(f"{Colors.YELLOW}⚠️ Drone already armed.{Colors.ENDC}")
            return
        print(f"{Colors.BLUE}🛡️ Arming drone…{Colors.ENDC}")
        for attempt in range(3):
            try:
                await self.drone.action.arm()
                self.armed = True
                print(f"{Colors.GREEN}✅ Armed successfully.{Colors.ENDC}")
                return
            except ActionError:
                print(f"{Colors.RED}❌ Arming failed (attempt {attempt+1}/3). Retrying…{Colors.ENDC}")
                await asyncio.sleep(1)
        raise ActionError("Failed to arm the drone after multiple attempts")

    async def disarm(self) -> None:
        """Disarm the drone, stopping offboard if necessary."""
        if not self.armed:
            print(f"{Colors.YELLOW}⚠️ Drone already disarmed.{Colors.ENDC}")
            return
        print(f"{Colors.BLUE}🛡️ Disarming drone…{Colors.ENDC}")
        if self.offboard_active:
            await self._stop_offboard()
        await self.drone.action.disarm()
        self.armed = False
        print(f"{Colors.GREEN}✅ Disarmed.{Colors.ENDC}")

    async def takeoff(self, altitude: float = 3.0) -> None:
        """Take off to a specified relative altitude."""
        if not self.armed:
            await self.arm()
        print(f"{Colors.BLUE}🚀 Taking off to {altitude:.1f} m…{Colors.ENDC}")
        await self.drone.action.set_takeoff_altitude(altitude)
        await self.drone.action.takeoff()
        timeout = time.time() + 30.0
        while time.time() < timeout:
            pos = await self.drone.telemetry.position().__anext__()
            if pos.relative_altitude_m >= altitude * 0.95:
                print(f"{Colors.GREEN}✅ Takeoff complete.{Colors.ENDC}")
                return
            await asyncio.sleep(0.2)
        raise ActionError("Takeoff timed out")

    async def land(self) -> None:
        """Land the drone and disarm once on the ground."""
        if self.offboard_active:
            await self._stop_offboard()
        print(f"{Colors.BLUE}🛬 Landing…{Colors.ENDC}")
        await self.drone.action.land()
        async for state in self.drone.telemetry.landed_state():
            if state == LandedState.ON_GROUND:
                break
            await asyncio.sleep(0.5)
        print(f"{Colors.GREEN}✅ Landed.{Colors.ENDC}")
        await self.disarm()

    # --- Motion primitives (relative to body frame) ---
    async def move_forward(self, distance: float, speed: float = 1.0) -> None:
        print(f"{Colors.BLUE}➡️ Moving forward {distance:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
        await self._move_body(distance, speed, 0.0, 0.0, 0.0)
        print(f"{Colors.GREEN}✅ Forward movement complete.{Colors.ENDC}")

    async def move_backward(self, distance: float, speed: float = 1.0) -> None:
        print(f"{Colors.BLUE}⬅️ Moving backward {distance:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
        await self._move_body(distance, -speed, 0.0, 0.0, 0.0)
        print(f"{Colors.GREEN}✅ Backward movement complete.{Colors.ENDC}")

    async def move_right(self, distance: float, speed: float = 1.0) -> None:
        print(f"{Colors.BLUE}➡️ Moving right {distance:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
        await self._move_body(distance, 0.0, speed, 0.0, 0.0)
        print(f"{Colors.GREEN}✅ Right movement complete.{Colors.ENDC}")

    async def move_left(self, distance: float, speed: float = 1.0) -> None:
        print(f"{Colors.BLUE}⬅️ Moving left {distance:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
        await self._move_body(distance, 0.0, -speed, 0.0, 0.0)
        print(f"{Colors.GREEN}✅ Left movement complete.{Colors.ENDC}")

    async def move_up(self, distance: float, speed: float = 1.0) -> None:
        print(f"{Colors.BLUE}⬆️ Ascending {distance:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
        await self._move_body(distance, 0.0, 0.0, -speed, 0.0)
        print(f"{Colors.GREEN}✅ Ascend complete.{Colors.ENDC}")

    async def move_down(self, distance: float, speed: float = 1.0) -> None:
        print(f"{Colors.BLUE}⬇️ Descending {distance:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
        await self._move_body(distance, 0.0, 0.0, speed, 0.0)
        print(f"{Colors.GREEN}✅ Descent complete.{Colors.ENDC}")

    async def yaw_right(self, angle_deg: float, yaw_rate: float = 30.0) -> None:
        print(f"{Colors.BLUE}🔄 Rotating clockwise {angle_deg:.1f}° at {yaw_rate:.1f}°/s…{Colors.ENDC}")
        duration = abs(angle_deg) / yaw_rate
        sign = 1.0 if angle_deg >= 0 else -1.0
        await self._move_body(duration, 0.0, 0.0, 0.0, sign * yaw_rate, override_duration=duration)
        print(f"{Colors.GREEN}✅ Rotation complete.{Colors.ENDC}")

    async def yaw_left(self, angle_deg: float, yaw_rate: float = 30.0) -> None:
        print(f"{Colors.BLUE}🔄 Rotating counter‑clockwise {angle_deg:.1f}° at {yaw_rate:.1f}°/s…{Colors.ENDC}")
        duration = abs(angle_deg) / yaw_rate
        sign = -1.0 if angle_deg >= 0 else 1.0
        await self._move_body(duration, 0.0, 0.0, 0.0, sign * yaw_rate, override_duration=duration)
        print(f"{Colors.GREEN}✅ Rotation complete.{Colors.ENDC}")

    # Aliases for yaw
    async def turn_cw(self, angle_deg: float, yaw_rate: float = 30.0) -> None:
        await self.yaw_right(angle_deg, yaw_rate)

    async def turn_ccw(self, angle_deg: float, yaw_rate: float = 30.0) -> None:
        await self.yaw_left(angle_deg, yaw_rate)

    async def fly_circle(self, radius: float, speed: float = 1.0, clockwise: bool = True) -> None:
        """Fly a horizontal circle using constant velocity and yaw rate."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        direction = "clockwise" if clockwise else "counter‑clockwise"
        print(f"{Colors.BLUE}🔃 Flying a {direction} circle of radius {radius:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
        omega_rad = speed / radius
        yaw_rate_deg = math.degrees(omega_rad)
        if not clockwise:
            yaw_rate_deg *= -1
        duration = (2 * math.pi * radius) / speed
        await self._move_body(duration, speed, 0.0, 0.0, yaw_rate_deg, override_duration=duration)
        print(f"{Colors.GREEN}✅ Circle complete.{Colors.ENDC}")

    async def hover(self, duration: float = 5.0) -> None:
        """Hold the current position for ``duration`` seconds."""
        print(f"{Colors.BLUE}🛑 Hovering for {duration:.1f} seconds…{Colors.ENDC}")
        await self._start_offboard()
        end_time = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_time:
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)
        print(f"{Colors.GREEN}✅ Hover complete.{Colors.ENDC}")

    async def square(self, side_length: float = 10.0, speed: float = 1.0) -> None:
        """Fly a square pattern with given side length and speed."""
        print(f"{Colors.BLUE}🟦 Flying square of {side_length:.1f} m sides at {speed:.1f} m/s…{Colors.ENDC}")
        for direction in (
            self.move_forward,
            self.move_right,
            self.move_backward,
            self.move_left,
        ):
            await direction(side_length, speed)
        print(f"{Colors.GREEN}✅ Square pattern complete.{Colors.ENDC}")

    # --- Navigation skills ---
    async def goto(self, lat: float, lon: float, alt: Optional[float] = None) -> None:
        if alt is None:
            pos = await self.drone.telemetry.position().__anext__()
            alt = pos.absolute_altitude_m
        att = await self.drone.telemetry.attitude_euler().__anext__()
        yaw_deg = att.yaw_deg
        if self.offboard_active:
            await self._stop_offboard()
        print(f"{Colors.BLUE}🎯 Going to {lat:.6f}, {lon:.6f} at {alt:.1f} m…{Colors.ENDC}")
        await self.drone.action.goto_location(lat, lon, alt, yaw_deg)
        print(f"{Colors.GREEN}✅ Goto command issued.{Colors.ENDC}")

    async def orbit(self, radius: float = 5.0, speed: float = 2.0, clockwise: bool = True,
                    center_lat: Optional[float] = None, center_lon: Optional[float] = None) -> None:
        if center_lat is None or center_lon is None:
            pos = await self.drone.telemetry.position().__anext__()
            center_lat = pos.latitude_deg
            center_lon = pos.longitude_deg
        pos = await self.drone.telemetry.position().__anext__()
        alt = pos.absolute_altitude_m
        if self.offboard_active:
            await self._stop_offboard()
        direction = "clockwise" if clockwise else "counter‑clockwise"
        print(f"{Colors.BLUE}🔄 Orbiting around {center_lat:.6f},{center_lon:.6f} (r={radius:.1f} m, v={speed:.1f} m/s, {direction})…{Colors.ENDC}")
        await self.drone.action.do_orbit(
            radius_m=radius,
            velocity_ms=speed,
            is_clockwise=clockwise,
            yaw_behavior=3,
            latitude_deg=center_lat,
            longitude_deg=center_lon,
            absolute_altitude_m=alt,
        )
        print(f"{Colors.GREEN}✅ Orbit started.{Colors.ENDC}")

    async def record_path(self, duration: float = 60.0) -> None:
        print(f"{Colors.BLUE}🔴 Recording path for {duration:.1f} s…{Colors.ENDC}")
        self.recorded_positions = []
        self.is_recording = True
        end_time = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_time and self.is_recording:
            pos = await self.drone.telemetry.position().__anext__()
            att = await self.drone.telemetry.attitude_euler().__anext__()
            self.recorded_positions.append(
                {
                    "lat": pos.latitude_deg,
                    "lon": pos.longitude_deg,
                    "alt": pos.absolute_altitude_m,
                    "yaw": att.yaw_deg,
                }
            )
            await asyncio.sleep(1.0)
        self.is_recording = False
        print(f"{Colors.GREEN}✅ Recorded {len(self.recorded_positions)} waypoints.{Colors.ENDC}")

    async def play_path(self, speed: float = 5.0) -> None:
        if not self.recorded_positions:
            raise RuntimeError("No recorded path to play")
        mission_items: List[MissionItem] = []
        for pos in self.recorded_positions:
            mission_items.append(
                MissionItem(
                    pos["lat"],
                    pos["lon"],
                    pos["alt"],
                    speed,
                    True,
                    float('nan'),
                    float('nan'),
                    0,
                    float('nan'),
                    float('nan'),
                    1.0,
                    pos["yaw"],
                    float('nan'),
                )
            )
        if self.offboard_active:
            await self._stop_offboard()
        print(f"{Colors.BLUE}▶️ Playing back recorded path ({len(mission_items)} waypoints) at {speed:.1f} m/s…{Colors.ENDC}")
        await self.drone.mission.set_return_to_launch_after_mission(True)
        await self.drone.mission.upload_mission(MissionPlan(mission_items))
        await self.drone.mission.start_mission()
        print(f"{Colors.GREEN}✅ Mission started.{Colors.ENDC}")

    async def fence_road(self, width: float = 20.0, length: float = 100.0) -> None:
        pos = await self.drone.telemetry.position().__anext__()
        att = await self.drone.telemetry.attitude_euler().__anext__()
        yaw_rad = math.radians(att.yaw_deg)
        lat = pos.latitude_deg
        lon = pos.longitude_deg
        half_width = width / 2
        half_length = length / 2
        lat_per_m = 1 / 111111.0
        lon_per_m = 1 / (111111.0 * math.cos(math.radians(lat)))
        fl_lat = lat + half_length * math.cos(yaw_rad) * lat_per_m - half_width * math.sin(yaw_rad) * lat_per_m
        fl_lon = lon + half_length * math.sin(yaw_rad) * lon_per_m + half_width * math.cos(yaw_rad) * lon_per_m
        fr_lat = lat + half_length * math.cos(yaw_rad) * lat_per_m + half_width * math.sin(yaw_rad) * lat_per_m
        fr_lon = lon + half_length * math.sin(yaw_rad) * lon_per_m - half_width * math.cos(yaw_rad) * lon_per_m
        br_lat = lat - half_length * math.cos(yaw_rad) * lat_per_m + half_width * math.sin(yaw_rad) * lat_per_m
        br_lon = lon - half_length * math.sin(yaw_rad) * lon_per_m - half_width * math.cos(yaw_rad) * lon_per_m
        bl_lat = lat - half_length * math.cos(yaw_rad) * lat_per_m - half_width * math.sin(yaw_rad) * lat_per_m
        bl_lon = lon - half_length * math.sin(yaw_rad) * lon_per_m + half_width * math.cos(yaw_rad) * lon_per_m
        polygon = [
            Point(fl_lat, fl_lon),
            Point(fr_lat, fr_lon),
            Point(br_lat, br_lon),
            Point(bl_lat, bl_lon),
        ]
        await self.drone.geofence.upload_geofence([polygon])
        print(f"{Colors.GREEN}✅ Geofence uploaded: road segment {width:.1f}×{length:.1f} m.{Colors.ENDC}")

    async def patrol(self, length: float = 50.0, passes: int = 2, speed: float = 5.0) -> None:
        pos = await self.drone.telemetry.position().__anext__()
        att = await self.drone.telemetry.attitude_euler().__anext__()
        yaw_rad = math.radians(att.yaw_deg)
        half_length = length / 2
        lat_per_m = 1 / 111111.0
        lon_per_m = 1 / (111111.0 * math.cos(math.radians(pos.latitude_deg)))
        forward_lat = pos.latitude_deg + half_length * math.cos(yaw_rad) * lat_per_m
        forward_lon = pos.longitude_deg + half_length * math.sin(yaw_rad) * lon_per_m
        backward_lat = pos.latitude_deg - half_length * math.cos(yaw_rad) * lat_per_m
        backward_lon = pos.longitude_deg - half_length * math.sin(yaw_rad) * lon_per_m
        mission_items: List[MissionItem] = []
        for _ in range(passes):
            mission_items.append(MissionItem(
                forward_lat,
                forward_lon,
                pos.absolute_altitude_m,
                speed,
                True,
                float('nan'),
                float('nan'),
                0,
                float('nan'),
                float('nan'),
                1.0,
                att.yaw_deg,
                float('nan'),
            ))
            mission_items.append(MissionItem(
                backward_lat,
                backward_lon,
                pos.absolute_altitude_m,
                speed,
                True,
                float('nan'),
                float('nan'),
                0,
                float('nan'),
                float('nan'),
                1.0,
                att.yaw_deg + 180.0,
                float('nan'),
            ))
        if self.offboard_active:
            await self._stop_offboard()
        print(f"{Colors.BLUE}🚓 Starting patrol: {length:.1f} m segment, {passes} passes at {speed:.1f} m/s…{Colors.ENDC}")
        await self.drone.mission.upload_mission(MissionPlan(mission_items))
        await self.drone.mission.start_mission()
        print(f"{Colors.GREEN}✅ Patrol mission started.{Colors.ENDC}")

    async def battery(self) -> None:
        batt = await self.drone.telemetry.battery().__anext__()
        print(f"{Colors.BLUE}🔋 Battery: {batt.remaining_percent * 100:.1f}% ({batt.voltage_v:.2f} V){Colors.ENDC}")

    async def status(self) -> None:
        pos = await self.drone.telemetry.position().__anext__()
        att = await self.drone.telemetry.attitude_euler().__anext__()
        battery = await self.drone.telemetry.battery().__anext__()
        flight_mode = await self.drone.telemetry.flight_mode().__anext__()
        status_str = (
            f"Latitude:  {pos.latitude_deg:.6f}\n"
            f"Longitude: {pos.longitude_deg:.6f}\n"
            f"Rel Alt:   {pos.relative_altitude_m:.2f} m\n"
            f"Abs Alt:   {pos.absolute_altitude_m:.2f} m\n"
            f"Roll:      {att.roll_deg:.2f}°\n"
            f"Pitch:     {att.pitch_deg:.2f}°\n"
            f"Yaw:       {att.yaw_deg:.2f}°\n"
            f"Battery:   {battery.remaining_percent * 100:.1f}% ({battery.voltage_v:.2f} V)\n"
            f"Mode:      {flight_mode}"
        )
        print(f"{Colors.BLUE}📡 Status:\n{status_str}{Colors.ENDC}")

    async def rtl(self) -> None:
        if self.offboard_active:
            await self._stop_offboard()
        print(f"{Colors.BLUE}↩️ Initiating Return‑to‑Launch…{Colors.ENDC}")
        await self.drone.action.return_to_launch()
        print(f"{Colors.GREEN}✅ RTL command issued. Drone is returning home.{Colors.ENDC}")

    async def pause_mission(self) -> None:
        print(f"{Colors.BLUE}⏸️ Pausing mission…{Colors.ENDC}")
        await self.drone.mission.pause_mission()
        print(f"{Colors.GREEN}✅ Mission paused. Drone is holding position.{Colors.ENDC}")

    async def resume_mission(self) -> None:
        print(f"{Colors.BLUE}▶️ Resuming mission…{Colors.ENDC}")
        await self.drone.mission.start_mission()
        print(f"{Colors.GREEN}✅ Mission resumed.{Colors.ENDC}")

    async def change_altitude(self, delta: float, speed: float = 1.0) -> None:
        if delta == 0:
            return
        if speed <= 0:
            raise ValueError("speed must be positive")
        if delta > 0:
            print(f"{Colors.BLUE}⬆️ Changing altitude by +{delta:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
            await self.move_up(abs(delta), speed)
        else:
            print(f"{Colors.BLUE}⬇️ Changing altitude by {delta:.1f} m at {speed:.1f} m/s…{Colors.ENDC}")
            await self.move_down(abs(delta), speed)

    # --- Offboard management ---
    async def _start_offboard(self) -> None:
        if self.offboard_active:
            return
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        try:
            await self.drone.offboard.start()
            self.offboard_active = True
        except OffboardError as exc:
            raise OffboardError(exc._result.result)

    async def _stop_offboard(self) -> None:
        if not self.offboard_active:
            return
        try:
            await self.drone.offboard.stop()
            self.offboard_active = False
        except OffboardError as exc:
            raise OffboardError(exc._result.result)

    async def _settle_hover(self, settle_time: float = 0.6) -> None:
        """Send zero velocity setpoints for a short period to stabilise after movement."""
        if not self.offboard_active:
            return
        end_time = asyncio.get_event_loop().time() + settle_time
        while asyncio.get_event_loop().time() < end_time:
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)

    async def _move_body(
        self,
        distance_or_duration: float,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
        override_duration: Optional[float] = None,
    ) -> None:
        """Internal helper to move using body‑frame velocity setpoints.

        This method streams body‑frame velocity commands at ~20 Hz for
        the computed duration.  At the end of the manoeuvre it
        continues to send zero velocity setpoints for a short settling
        period using :py:meth:`_settle_hover`, then stops offboard
        mode to hand control back to PX4.
        """
        if override_duration is not None:
            duration = override_duration
        else:
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            if speed == 0.0:
                return
            duration = abs(distance_or_duration) / speed
        await self._start_offboard()
        end_time = asyncio.get_event_loop().time() + duration
        try:
            while asyncio.get_event_loop().time() < end_time:
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(vx, vy, vz, yaw_rate)
                )
                await asyncio.sleep(0.05)
        finally:
            await self._settle_hover(0.6)
            await self._stop_offboard()


async def main_cli() -> None:
    controller = DroneAgentController()
    print(f"{Colors.BLUE}🔌 Connecting to drone and arming…{Colors.ENDC}")
    await controller.connect()
    await controller.arm()
    print(f"{Colors.GREEN}🤖 Drone ready. Type 'help' for commands.{Colors.ENDC}")
    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, "\nagent> "
            )
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break
        parts = user_input.strip().split()
        if not parts:
            continue
        cmd, *args = parts
        cmd = cmd.lower()
        try:
            if cmd == "help":
                print(
                    """
Available commands:
  takeoff [alt]               – Take off to a relative altitude
  land                        – Land and disarm
  forward d [s]               – Move forward by d metres at optional speed s
  backward d [s]              – Move backward by d metres
  right d [s]                 – Move right by d metres
  left d [s]                  – Move left by d metres
  up d [s]                    – Ascend by d metres
  down d [s]                  – Descend by d metres
  yaw_left a [rate]           – Rotate CCW by a degrees
  yaw_right a [rate]          – Rotate CW by a degrees
  turn_ccw a [rate]           – Alias for yaw_left
  turn_cw a [rate]            – Alias for yaw_right
  circle r [speed] [cw|ccw]   – Fly circle of radius r at speed
  goto lat lon [alt]          – Fly to global coordinates
  orbit [r] [speed] [cw|ccw]  – Orbit around current position
  hover [duration]            – Hold position
  square [size] [speed]       – Fly a square
  record [seconds]            – Record path
  play [speed]                – Replay recorded path
  fence [width] [length]      – Set geofence along road
  patrol [length] [passes]    – Patrol along a line
  battery                     – Print battery status
  status                      – Print telemetry snapshot
  rtl                         – Return to launch and land
  pause                       – Pause mission
  resume                      – Resume mission
  alt delta [speed]           – Change altitude by delta metres
  exit                        – Exit the program
"""
                )
            elif cmd == "takeoff":
                alt = float(args[0]) if args else 3.0
                await controller.takeoff(alt)
            elif cmd == "land":
                await controller.land()
            elif cmd == "forward":
                dist = float(args[0]); spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.move_forward(dist, spd)
            elif cmd == "backward":
                dist = float(args[0]); spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.move_backward(dist, spd)
            elif cmd == "right":
                dist = float(args[0]); spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.move_right(dist, spd)
            elif cmd == "left":
                dist = float(args[0]); spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.move_left(dist, spd)
            elif cmd == "up":
                dist = float(args[0]); spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.move_up(dist, spd)
            elif cmd == "down":
                dist = float(args[0]); spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.move_down(dist, spd)
            elif cmd in ("yaw_left", "turn_ccw"):
                ang = float(args[0]); rate = float(args[1]) if len(args) > 1 else 30.0
                await controller.yaw_left(ang, rate)
            elif cmd in ("yaw_right", "turn_cw"):
                ang = float(args[0]); rate = float(args[1]) if len(args) > 1 else 30.0
                await controller.yaw_right(ang, rate)
            elif cmd == "circle":
                r = float(args[0])
                spd = float(args[1]) if len(args) > 1 else 1.0
                cw = True
                if len(args) > 2 and args[2].lower() in ("ccw", "counter", "counterclockwise"):
                    cw = False
                await controller.fly_circle(r, spd, cw)
            elif cmd == "goto":
                if len(args) < 2:
                    print("Usage: goto <lat> <lon> [alt]")
                    continue
                lat = float(args[0]); lon = float(args[1])
                alt = float(args[2]) if len(args) > 2 else None
                await controller.goto(lat, lon, alt)
            elif cmd == "orbit":
                r = float(args[0]) if len(args) > 0 else 5.0
                spd = float(args[1]) if len(args) > 1 else 2.0
                cw = True
                if len(args) > 2 and args[2].lower() in ("ccw", "counter", "counterclockwise"):
                    cw = False
                await controller.orbit(r, spd, cw)
            elif cmd == "hover":
                dur = float(args[0]) if args else 5.0
                await controller.hover(dur)
            elif cmd == "square":
                size = float(args[0]) if args else 10.0
                spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.square(size, spd)
            elif cmd == "record":
                dur = float(args[0]) if args else 60.0
                await controller.record_path(dur)
            elif cmd == "play":
                spd = float(args[0]) if args else 5.0
                await controller.play_path(spd)
            elif cmd == "fence":
                width = float(args[0]) if len(args) > 0 else 20.0
                length = float(args[1]) if len(args) > 1 else 100.0
                await controller.fence_road(width, length)
            elif cmd == "patrol":
                length = float(args[0]) if len(args) > 0 else 50.0
                passes = int(args[1]) if len(args) > 1 else 2
                await controller.patrol(length, passes)
            elif cmd == "battery":
                await controller.battery()
            elif cmd == "status":
                await controller.status()
            elif cmd == "rtl":
                await controller.rtl()
            elif cmd == "pause":
                await controller.pause_mission()
            elif cmd == "resume":
                await controller.resume_mission()
            elif cmd == "alt":
                if not args:
                    print("Usage: alt <delta> [speed]")
                    continue
                delta = float(args[0])
                spd = float(args[1]) if len(args) > 1 else 1.0
                await controller.change_altitude(delta, spd)
            elif cmd == "exit":
                break
            else:
                print(f"Unknown command: {cmd}")
        except Exception as exc:
            print(f"Error: {exc}")
    try:
        await controller.land()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main_cli())
    except KeyboardInterrupt:
        pass