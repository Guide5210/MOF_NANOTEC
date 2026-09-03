# -*- coding: utf-8 -*-
"""เลือกพอร์ตด้วย VID:PID — ห้ามคว้าพอร์ตของจอมาเป็น CONTROL เด็ดขาด"""
import unittest
from _helpers import ROOT, FakePort  # noqa
from ecstation.core import ports as P

CONTROL = FakePort("COM3", "USB-SERIAL CH340 (COM3)", "USB VID:PID=1A86:7523")
P4_LOG  = FakePort("COM5", "USB-Enhanced-SERIAL CH343 (COM5)", "USB VID:PID=1A86:55D3")
P4_JTAG = FakePort("COM7", "USB Serial Device (COM7)", "USB VID:PID=303A:1001 SER=x")
P4_JTAG2 = FakePort("COM8", "USB JTAG/serial debug unit (COM8)", "USB VID:PID=303A:1001")
OTHER   = FakePort("COM9", "USB Serial Device (COM9)", "USB VID:PID=0403:6001")


def ports(*ps):
    return lambda: list(ps)


class TestControlRole(unittest.TestCase):
    def test_picks_control_when_all_present(self):
        self.assertEqual(P.find(P.ROLE_CONTROL, ports(P4_JTAG, P4_LOG, CONTROL)), "COM3")

    def test_never_picks_p4_jtag(self):
        """เคสที่เคยพัง: ถอด CONTROL ออกแล้วไปคว้าพอร์ต NDJSON ของจอ"""
        self.assertIsNone(P.find(P.ROLE_CONTROL, ports(P4_JTAG, P4_LOG)))
        self.assertIsNone(P.find(P.ROLE_CONTROL, ports(P4_JTAG)))
        self.assertIsNone(P.find(P.ROLE_CONTROL, ports(P4_JTAG2)))

    def test_jtag_by_name_also_excluded(self):
        self.assertEqual(P.find(P.ROLE_CONTROL, ports(P4_JTAG2, CONTROL)), "COM3")

    def test_generic_usb_still_allowed_as_last_resort(self):
        self.assertEqual(P.find(P.ROLE_CONTROL, ports(OTHER)), "COM9")

    def test_no_ports(self):
        self.assertIsNone(P.find(P.ROLE_CONTROL, ports()))


class TestBridgeRole(unittest.TestCase):
    def test_only_exact_vid_pid(self):
        self.assertEqual(P.find(P.ROLE_P4_BRIDGE, ports(CONTROL, P4_LOG, P4_JTAG)), "COM7")

    def test_never_guesses(self):
        self.assertIsNone(P.find(P.ROLE_P4_BRIDGE, ports(CONTROL, P4_LOG, OTHER)))

    def test_control_port_never_used_as_bridge(self):
        self.assertIsNone(P.find(P.ROLE_P4_BRIDGE, ports(CONTROL)))


class TestDescribe(unittest.TestCase):
    def test_roles_named(self):
        rows = {r["device"]: r["role"] for r in
                P.describe(ports(CONTROL, P4_LOG, P4_JTAG, OTHER))}
        self.assertIn("CONTROL", rows["COM3"])
        self.assertIn("log/flash", rows["COM5"])
        self.assertIn("bridge", rows["COM7"])
        self.assertEqual(rows["COM9"], "—")


if __name__ == "__main__":
    unittest.main(verbosity=2)
