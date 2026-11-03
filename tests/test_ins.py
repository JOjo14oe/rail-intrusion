import math
from pathlib import Path

from railintrude.ins import forward_speed_ms, read_ins, read_ins_speeds, speed_for_frame

_CSV = (
    "frame_idx,timestamp,latitude,longitude,height,north_velocity,east_velocity,"
    "up_velocity,roll,pitch,azimuth\n"
    "250,1631531385.100000000,53.552400582617,10.026907234934,53.327156103216,"
    "1.254061594160,-5.181509173141,0.007452970897,-2.060456663868,0.213799218915,"
    "283.519299649580\n"
)


def test_forward_speed_is_zero_when_moving_sideways():
    # azimuth 0 == heading north; a pure east velocity is then pure lateral motion
    assert abs(forward_speed_ms(0.0, 5.0, 0.0)) < 1e-9


def test_forward_speed_matches_heading_when_driving_north():
    assert abs(forward_speed_ms(5.0, 0.0, 0.0) - 5.0) < 1e-9


def test_forward_speed_rotates_with_azimuth():
    # real row from OSDaR23 2_station_berliner_tor_2.1 frame 250: heading 283.5 deg
    n, e = 1.254061594160, -5.181509173141
    v = forward_speed_ms(n, e, 283.519299649580)
    # this row travels nearly along its own heading, so forward == |v| to 1 cm/s
    assert abs(v - math.hypot(n, e)) < 0.01
    # and the lateral component is ~0, which is what the rotation is for
    lateral = -n * math.sin(math.radians(283.519299649580)) + e * math.cos(
        math.radians(283.519299649580)
    )
    assert abs(lateral) < 0.01


def test_read_ins_parses_one_csv(tmp_path):
    p = tmp_path / "250_1631531385.100000000.csv"
    p.write_text(_CSV, encoding="utf-8")
    rec = read_ins(p)
    assert rec["frame_idx"] == 250
    assert abs(rec["north_velocity"] - 1.254061594160) < 1e-9
    assert abs(rec["azimuth"] - 283.519299649580) < 1e-9


def test_read_ins_speeds_keys_by_frame_and_reports_kmh(tmp_path):
    p = tmp_path / "250_1631531385.100000000.csv"
    p.write_text(_CSV, encoding="utf-8")
    table = read_ins_speeds(tmp_path)
    assert list(table) == [250]
    assert abs(table[250] - math.hypot(1.254061594160, 5.181509173141) * 3.6) < 0.04


def test_speed_for_frame_uses_nearest_when_exact_missing():
    table = {250: 19.2, 252: 20.0}
    assert speed_for_frame(table, 250) == 19.2
    assert speed_for_frame(table, 251) == 19.2
    assert speed_for_frame(table, 999) == 20.0
    assert speed_for_frame({}, 250) is None


def test_read_ins_speeds_ignores_bad_rows(tmp_path):
    bad = tmp_path / "1_1.0.csv"
    bad.write_text("frame_idx,timestamp\n1,1.0\n", encoding="utf-8")
    assert read_ins_speeds(tmp_path) == {}
