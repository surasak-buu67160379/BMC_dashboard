"""VoltWise AI — สร้างชุดข้อมูลจำลอง 6 เดือน (3 เดือนก่อนใช้ AI + 3 เดือนหลังใช้ AI)

หลักคิด
- ไซต์เดียวกัน ฮาร์ดแวร์เดิม (โซลาร์ + BESS) ช่วง "ก่อน" แบตวิ่งตามกฎของผู้ผลิต (rule-based)
  ช่วง "หลัง" แบตถูกคุมด้วย AI (forecast-aware)
- ทั้งสองกลยุทธ์รันคู่กันตลอด 6 เดือน => ช่วง "หลัง" มีเส้น "ถ้าไม่มี AI" ของวันเดียวกัน
  เทียบได้ทั้ง (ก่อน vs หลัง) และ (หลัง vs ถ้าไม่มี AI)
- ผลลัพธ์ปล่อยตามจริง ไม่ปรับให้ตรงตัวเลขในสไลด์ สมมติฐานทั้งหมดอยู่ในบล็อกด้านล่าง
"""
import numpy as np
import pandas as pd

# ====================== สมมติฐาน (แก้ที่นี่ที่เดียว) ======================
SEED = 42
START, END, AI_START = "2026-04-01", "2026-09-30 23:00:00", "2026-07-01"
MATCH_SEASONS = True       # True = สองครึ่งมีโปรไฟล์ฤดูกาลเหมือนกัน (เทียบแฟร์) / False = ฤดูกาลตามปฏิทินเดิม

BATT_KWH, BATT_KW, EFF_RT = 1000.0, 250.0, 0.90      # ความจุ, กำลังสูงสุด, ประสิทธิภาพ round-trip
SOC_MIN, SOC_MAX = 10.0, 95.0
ETA = EFF_RT ** 0.5                                   # ประสิทธิภาพขาชาร์จ = ขาจ่าย

TARIFF_ON, TARIFF_OFF = 4.32, 2.63                    # บาท/kWh (On-Peak = จันทร์-ศุกร์ 09:00-22:00)
DEMAND_THB_PER_KW = 132.93                            # สมมติ: ค่า Demand Charge บาท/kW/เดือน — ตรวจกับอัตราจริงก่อนใช้

LEGACY = dict(trigger_kw=650, target_kw=600, soc_floor=15, charge_kw=200)   # กฎเดิมของคุณ
SMART = dict(soc_floor=20, reserve_pct=3.0, fc_day=0.05, fc_hr=0.03)       # floor 20% = กันไว้รับไฟวูบ, พยากรณ์คลาด ~5%
SAG = dict(p_base=0.004, p_cloud=0.05, drop=(0.30, 0.70),                  # ไฟวูบจากแดดวูบ (วันทำงาน 08-16 น.)
           rt_legacy=0.30, rt_ai=0.95,                                      # สมมติ: โอกาสซับไฟวูบสำเร็จ — ต้องวัดจริงใน Pilot
           minutes=(5, 30), loss_thb_min=1000)                              # หยุดไลน์กี่นาที / เสียหายบาทต่อนาที (สมมติ)

rng = np.random.default_rng(SEED)

# ====================== 1) โครงเวลา + ฤดูกาล ======================
df = pd.DataFrame({"timestamp": pd.date_range(START, END, freq="1h")})
n, nd = len(df), len(df) // 24
df["month"] = df["timestamp"].dt.strftime("%Y-%m")
df["month_name"] = df["timestamp"].dt.strftime("%b %Y")
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["phase"] = np.where(df["timestamp"] >= AI_START, "after", "before")
hr, wk = df["hour"].to_numpy(), (df["dayofweek"] < 5).to_numpy()
onp = wk & (hr >= 9) & (hr < 22)
mo = df["timestamp"].dt.month.to_numpy()

if MATCH_SEASONS:   # เดือนที่ 1/2/3 ของแต่ละครึ่งใช้ตัวคูณชุดเดียวกัน
    ph = (mo - 4) % 3
    s_solar, s_load = np.array([1.10, 1.00, 0.90])[ph], np.array([1.05, 1.00, 0.97])[ph]
else:
    s_solar = np.where(np.isin(mo, [4, 5]), 1.15, np.where(np.isin(mo, [7, 8]), 0.85, 1.0))
    s_load = np.where(np.isin(mo, [4, 5]), 1.10, np.where(np.isin(mo, [7, 8]), 0.95, 1.0))

# ====================== 2) โหลด โซลาร์ ไฟวูบ ======================
cloud = np.repeat(0.55 + 0.45 * rng.beta(5, 2, nd), 24)       # ความใสของท้องฟ้ารายวัน
load_d = np.repeat(rng.normal(1.0, 0.05, nd), 24)             # โหลดแต่ละวันไม่เท่ากัน -> พีคแต่ละวันต่างกัน
load = np.where(wk, 850 + 350 * np.sin((hr - 8) * np.pi / 12).clip(0) + rng.normal(0, 40, n),
                350 + rng.normal(0, 25, n))
load = (load * s_load * load_d).clip(200)
solar = ((520 * np.sin((hr - 6) * np.pi / 12).clip(0) + rng.normal(0, 20, n)).clip(0)) * s_solar * cloud

day = wk & (hr >= 8) & (hr <= 16)
sag = rng.random(n) < np.where(day, SAG["p_base"] + SAG["p_cloud"] * (1 - cloud), 0)
sag_kw = np.where(sag, solar * rng.uniform(*SAG["drop"], n), 0)
solar = solar - sag_kw
sag_down_min = rng.uniform(*SAG["minutes"], n)               # ใช้ค่าเดียวกันทั้งสองกลยุทธ์ (จับคู่เทียบ)

net = load - solar
tariff = np.where(onp, TARIFF_ON, TARIFF_OFF)


# ====================== 3) ตัวควบคุมแบต ======================
def step(soc, a):
    """a>0 ชาร์จ (กิน kW จากกริด/โซลาร์), a<0 จ่าย; คืน SoC ใหม่ (คิดประสิทธิภาพ)"""
    return soc + (a * ETA if a > 0 else a / ETA) / BATT_KWH * 100


def legacy_ctrl():
    """กฎเดิม: ชาร์จนอก Peak จนเต็ม, ช่วง Peak จ่ายทันทีที่ net > 650 kW (แก้: ไม่จ่ายเกินที่มี)"""
    L, soc = LEGACY, 50.0
    act, socs = np.zeros(n), np.zeros(n)
    for i in range(n):
        a = 0.0
        if onp[i]:
            if net[i] > L["trigger_kw"] and soc > L["soc_floor"]:
                a = -min(BATT_KW, net[i] - L["target_kw"], (soc - SOC_MIN) / 100 * BATT_KWH * ETA)
        elif soc < 95.0:
            a = min(L["charge_kw"], (100 - soc) * 20, (SOC_MAX - soc) / 100 * BATT_KWH / ETA)
        soc = step(soc, a)
        act[i], socs[i] = a, soc
    return act, socs


def smart_ctrl():
    """AI: เช้าพยากรณ์ net load ช่วง Peak -> หา "เพดาน" ที่แบตตัดไหวทั้งวัน -> ชาร์จเท่าที่จำเป็นแบบเฉลี่ย
    -> ช่วง Peak จ่ายเฉพาะส่วนที่เกินเพดาน (ใช้ค่าจริง ถ้าพยากรณ์คลาดก็พลาดได้) และเก็บ SoC กันไฟวูบไว้"""
    S, soc = SMART, 50.0
    act, socs = np.zeros(n), np.zeros(n)
    for d in range(nd):
        nt, op = net[d * 24:(d + 1) * 24], onp[d * 24:(d + 1) * 24]
        cap, target = np.inf, SOC_MIN
        if op.any():
            f = nt * (1 + rng.normal(0, S["fc_day"]) + rng.normal(0, S["fc_hr"], 24))
            reach = min(SOC_MAX, soc + 9 * BATT_KW * ETA / BATT_KWH * 100)
            avail = (reach - S["soc_floor"]) / 100 * BATT_KWH * ETA
            need = lambda c: np.minimum(np.maximum(f[op] - c, 0), BATT_KW).sum()
            lo, hi = 0.0, f[op].max()
            if need(0.0) <= avail:
                cap = 0.0
            else:
                for _ in range(40):
                    mid = (lo + hi) / 2
                    lo, hi = (lo, mid) if need(mid) <= avail else (mid, hi)
                cap = hi
            target = min(SOC_MAX, S["soc_floor"] + need(cap) / ETA / BATT_KWH * 100 + S["reserve_pct"])
        for h in range(24):
            i, a = d * 24 + h, 0.0
            if (not op[h]) and h < 9 and soc < target:
                a = min(BATT_KW, (target - soc) / 100 * BATT_KWH / ETA / (9 - h))
            elif op[h] and nt[h] > cap and soc > S["soc_floor"]:
                a = -min(BATT_KW, nt[h] - cap, (soc - S["soc_floor"]) / 100 * BATT_KWH * ETA)
            soc = step(soc, a)
            act[i], socs[i] = a, soc
    return act, socs


a_leg, s_leg = legacy_ctrl()
a_ai, s_ai = smart_ctrl()

# ====================== 4) ผลลัพธ์รายชั่วโมง ======================
df["factory_load_kw"], df["solar_gen_kw"], df["tou_tariff_thb"] = load, solar, tariff
df["nobatt_grid_kw"] = net.clip(0)                               # อ้างอิง: ไม่มีแบต
df["baseline_battery_action_kw"], df["baseline_soc_pct"] = a_leg, s_leg
df["baseline_grid_kw"] = (net + a_leg).clip(0)                   # "ไม่ใช้ AI" (กฎผู้ผลิต)
df["baseline_cost_thb"] = df["baseline_grid_kw"] * tariff
df["ai_battery_action_kw"], df["battery_soc_pct"] = a_ai, s_ai
df["ai_grid_kw"] = (net + a_ai).clip(0)                          # "ใช้ AI"
df["ai_cost_thb"] = df["ai_grid_kw"] * tariff
df["savings_thb"] = df["baseline_cost_thb"] - df["ai_cost_thb"]  # AI เทียบไม่มี AI (ค่าจริงเฉพาะช่วง after)
after = (df["phase"] == "after").to_numpy()
df["actual_grid_kw"] = np.where(after, df["ai_grid_kw"], df["baseline_grid_kw"])
df["actual_cost_thb"] = np.where(after, df["ai_cost_thb"], df["baseline_cost_thb"])
df["actual_soc_pct"] = np.where(after, s_ai, s_leg)
df["co2_avoided_kg"] = solar * 0.4999

# ไฟวูบ: ซับได้ถ้าแบตมี SoC พอ และสุ่มผ่านโอกาสซับ (สมมติฐานใน SAG)
prev = lambda s, init=50.0: np.r_[init, s[:-1]]
rt_leg = sag & (prev(s_leg) >= 20) & (rng.random(n) < SAG["rt_legacy"])
rt_ai = sag & (prev(s_ai) >= SMART["soc_floor"]) & (rng.random(n) < SAG["rt_ai"])
df["sag_event"], df["sag_kw_drop"] = sag.astype(int), sag_kw
df["ride_through_baseline"], df["ride_through_ai"] = rt_leg.astype(int), rt_ai.astype(int)
df["sag_loss_baseline_thb"] = np.where(sag & ~rt_leg, sag_down_min * SAG["loss_thb_min"], 0)
df["sag_loss_ai_thb"] = np.where(sag & ~rt_ai, sag_down_min * SAG["loss_thb_min"], 0)
df["sag_loss_actual_thb"] = np.where(after, df["sag_loss_ai_thb"], df["sag_loss_baseline_thb"])

# ====================== 5) ตรวจความถูกต้อง ======================
assert (df[["baseline_grid_kw", "ai_grid_kw"]] >= 0).all().all()
assert min(s_leg.min(), s_ai.min()) >= SOC_MIN - 1e-6 and max(s_leg.max(), s_ai.max()) <= SOC_MAX + 1e-6
assert max(np.abs(a_leg).max(), np.abs(a_ai).max()) <= BATT_KW + 1e-6
for a, s in ((a_leg, s_leg), (a_ai, s_ai)):   # พลังงานในแบตต้องสอดคล้องกับที่สั่งชาร์จ/จ่าย
    stored = np.where(a > 0, a * ETA, a / ETA).sum() / BATT_KWH * 100
    assert abs((s[-1] - 50.0) - stored) < 1e-6

# ====================== 6) สรุปรายเดือน (รวม Demand Charge) ======================
def monthly(prefix):
    g = df.assign(g=df[f"{prefix}_grid_kw"], c=df[f"{prefix}_cost_thb"]).groupby("month", sort=True)
    out = pd.DataFrame({"energy_thb": g["c"].sum(),
                        "peak_kw": g.apply(lambda x: x.loc[onp[x.index], "g"].max(), include_groups=False)})
    out["demand_thb"] = out["peak_kw"] * DEMAND_THB_PER_KW
    out["bill_thb"] = out["energy_thb"] + out["demand_thb"]
    return out

m_leg, m_ai = monthly("baseline"), monthly("ai")
summ = pd.concat({"baseline": m_leg, "ai": m_ai}, axis=1)
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ["phase"] = np.where(summ.index >= AI_START[:7], "after", "before")
summ["actual_bill_thb"] = np.where(summ["phase"] == "after", summ["ai_bill_thb"], summ["baseline_bill_thb"])
summ["ai_saving_pct"] = 100 * (1 - summ["ai_bill_thb"] / summ["baseline_bill_thb"])
summ["sag_loss_actual_thb"] = df.groupby("month")["sag_loss_actual_thb"].sum()
summ["sag_events"] = df.groupby("month")["sag_event"].sum()

df.to_csv("voltwise_6months_data.csv", index=False)
summ.round(1).to_csv("voltwise_6months_monthly.csv")

# ====================== 7) พิมพ์ผลให้ตรวจ ======================
pd.set_option("display.width", 200)
print("== รายเดือน (บาท) ==")
print(summ[["phase", "baseline_peak_kw", "ai_peak_kw", "baseline_bill_thb", "ai_bill_thb", "ai_saving_pct",
            "sag_events", "sag_loss_actual_thb"]].round(1))
aft = summ[summ.phase == "after"]
print(f"\nช่วงหลัง: AI ลดบิลรวม (พลังงาน+Demand) {100 * (1 - aft.ai_bill_thb.sum() / aft.baseline_bill_thb.sum()):.1f}%"
      f" | ลด Peak เฉลี่ย {(aft.baseline_peak_kw - aft.ai_peak_kw).mean():.0f} kW")
bef, aft_d = df[~after], df[after]
b = lambda x, c: (x[c] < 0).sum()
for name, d in (("baseline", df[after]), ("ai", df[after])):
    p = "baseline" if name == "baseline" else "battery"
    soc, act = d["baseline_soc_pct" if name == "baseline" else "battery_soc_pct"], d[f"{name}_battery_action_kw"]
    print(f"แบต[{name:8s}] รอบเต็มเทียบเท่า/เดือน {(-act.clip(upper=0)).sum() / BATT_KWH / 3:5.1f} | "
          f"SoC เฉลี่ย {soc.mean():4.1f}% | ชม.ที่ SoC>90% {(soc > 90).sum() / 3:5.0f}/เดือน")
print(f"ไฟวูบช่วงหลัง: {int(aft_d.sag_event.sum())} ครั้ง | ซับสำเร็จ baseline {int(aft_d.ride_through_baseline.sum())}"
      f" vs AI {int(aft_d.ride_through_ai.sum())} | เสียหาย baseline {aft_d.sag_loss_baseline_thb.sum():,.0f}"
      f" vs AI {aft_d.sag_loss_ai_thb.sum():,.0f} บาท")
print("\n✅ สร้าง voltwise_6months_data.csv และ voltwise_6months_monthly.csv แล้ว")