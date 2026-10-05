"""VoltWise — KPI Dashboard (Streamlit)
รัน:  pip install streamlit pandas plotly
      streamlit run app.py
ต้องมีไฟล์ voltwise_6months_data.csv และ voltwise_6months_monthly.csv อยู่โฟลเดอร์เดียวกัน
(ไฟล์สร้างข้อมูลเดิมให้เปลี่ยนชื่อเป็น generate_data.py)
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="VoltWise KPI", page_icon="⚡", layout="wide")

BAD, OK, GRAY, SOC = "#d93a3f", "#0e7c6b", "#8a97a0", "#f0b44c"
SOC_FLOOR_LEGACY = 15


@st.cache_data
def load():
    d = pd.read_csv("voltwise_6months_data.csv", parse_dates=["timestamp"])
    m = pd.read_csv("voltwise_6months_monthly.csv")
    return d, m


d, m = load()
d["onpeak"] = (d["dayofweek"] < 5) & (d["hour"] >= 9) & (d["hour"] < 22)
bef_m, aft_m = m[m.phase == "before"], m[m.phase == "after"]
bef, aft = d[d.phase == "before"], d[d.phase == "after"]
f = lambda x: f"{x:,.0f}"

# ---------------- หัวเรื่อง ----------------
st.title("แบตเตอรี่ 1 MWh ติดตั้งแล้ว แต่ยังไม่ช่วยลดค่าไฟช่วงพีค")
st.caption("VoltWise · โซลาร์ + BESS 1,000 kWh / 250 kW · ข้อมูลจำลองรายชั่วโมง เม.ย.–ก.ย. 2026")
st.error("ระบบกฎเดิมปล่อยให้แบตหมดตั้งแต่บ่าย ทำให้ช่วงเย็นที่ไฟพีคที่สุดไม่มีแบตช่วย "
         "เดือนที่พีคสูงสุดจึงเท่ากับไม่มีแบตเลย")

# ---------------- กราฟหลัก ----------------
st.subheader("ปัญหาหลัก: แบตหมดก่อนไฟพีค")
st.caption("ค่าเฉลี่ยวันทำงาน ช่วงก่อนใช้ AI (ระบบกฎเดิม)")
p = bef[bef.dayofweek < 5].groupby("hour")[["nobatt_grid_kw", "baseline_grid_kw", "baseline_soc_pct"]].mean()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_vrect(x0=9, x1=22, fillcolor="gray", opacity=0.08, line_width=0,
              annotation_text="On-Peak 09–22 น.", annotation_position="top left")
fig.add_trace(go.Scatter(x=p.index, y=p.baseline_soc_pct, name="ประจุแบต % (แกนขวา)", fill="tozeroy",
                         line=dict(width=0), fillcolor="rgba(240,180,76,.35)"), secondary_y=True)
fig.add_trace(go.Scatter(x=p.index, y=p.nobatt_grid_kw, name="ไฟจากกริดถ้าไม่มีแบต",
                         line=dict(color=GRAY, dash="dash")), secondary_y=False)
fig.add_trace(go.Scatter(x=p.index, y=p.baseline_grid_kw, name="ไฟจากกริดจริง (กฎเดิม)",
                         line=dict(color=BAD, width=3)), secondary_y=False)
pk_h = p.baseline_grid_kw[9:22].idxmax()
fig.add_annotation(x=pk_h, y=p.baseline_grid_kw[pk_h], text=f"พีค {f(p.baseline_grid_kw[pk_h])} kW",
                   showarrow=True, arrowhead=2, ax=40, ay=-30, font=dict(color=BAD))
fig.update_yaxes(title_text="kW", secondary_y=False)
fig.update_yaxes(title_text="SoC %", range=[0, 100], secondary_y=True, showgrid=False)
fig.update_xaxes(title_text="ชั่วโมง", dtick=3)
fig.update_layout(height=380, margin=dict(t=30, b=10), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, width="stretch")

# ---------------- KPI ปัญหา ----------------
st.subheader("KPI ของปัญหา (ก่อนใช้ AI)")
op = bef[bef.onpeak]
sag_b = int(bef.sag_event.sum())
c = st.columns(4)
c[0].metric("พีคสูงสุดของเดือน", f"{f(bef_m.baseline_peak_kw.max())} kW",
            f"แบตลดจากไม่มีแบตเพียง {f(bef.nobatt_grid_kw.max() - bef_m.baseline_peak_kw.max())} kW",
            delta_color="off")
c[1].metric("Demand Charge เฉลี่ย/เดือน", f"{f(bef_m.baseline_demand_thb.mean())} บาท")
c[2].metric("ชั่วโมง On-Peak ที่แบตแตะขั้นต่ำ", f"{100 * (op.baseline_soc_pct <= SOC_FLOOR_LEGACY).mean():.0f}%",
            f"SoC ≤ {SOC_FLOOR_LEGACY}%", delta_color="off")
c[3].metric("ไฟวูบจากโซลาร์", f"{sag_b} ครั้ง",
            f"ซับได้ {int(bef.ride_through_baseline.sum())} · เสียหาย {f(bef.sag_loss_baseline_thb.sum())} บาท",
            delta_color="off")

# ---------------- พีครายเดือน ----------------
st.subheader("พีคสูงสุดรายเดือน (kW) ตัวกำหนดค่า Demand Charge")
st.caption("แท่ง = ที่เกิดขึ้นจริง · เครื่องหมาย ◆ = ถ้าไม่ใช้ AI ในเดือนเดียวกัน")
is_aft = m.phase == "after"
actual = m.baseline_peak_kw.where(~is_aft, m.ai_peak_kw)
fig2 = go.Figure()
fig2.add_bar(x=m.month, y=actual, marker_color=[OK if a else BAD for a in is_aft],
             text=actual.round(0), textposition="outside", name="จริง")
fig2.add_scatter(x=m.month[is_aft], y=m.baseline_peak_kw[is_aft], mode="markers",
                 marker=dict(symbol="diamond", size=12, color="black"), name="ถ้าไม่มี AI")
fig2.update_yaxes(range=[600, 1300], title_text="kW")
fig2.update_layout(height=340, margin=dict(t=20, b=10), showlegend=True, legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig2, width="stretch")

# ---------------- ก่อน vs หลัง ----------------
st.subheader("หลังใช้ AI เปลี่ยนไปเท่าไร (ก.ค.–ก.ย.)")
b_en, a_en = (aft_m.baseline_bill_thb - aft_m.baseline_demand_thb).sum(), (aft_m.ai_bill_thb - aft_m.ai_demand_thb).sum()
rows = [
    ("พีคเฉลี่ย/เดือน (kW)", aft_m.baseline_peak_kw.mean(), aft_m.ai_peak_kw.mean(), True),
    ("Demand Charge (บาท)", aft_m.baseline_demand_thb.sum(), aft_m.ai_demand_thb.sum(), True),
    ("ค่าพลังงาน (บาท)", b_en, a_en, True),
    ("บิลรวม (บาท)", aft_m.baseline_bill_thb.sum(), aft_m.ai_bill_thb.sum(), True),
    ("ความเสียหายจากไฟวูบ (บาท)", aft.sag_loss_baseline_thb.sum(), aft.sag_loss_ai_thb.sum(), True),
    (f"ไฟวูบที่ซับได้ (ครั้งจาก {int(aft.sag_event.sum())})", aft.ride_through_baseline.sum(),
     aft.ride_through_ai.sum(), False),
]
t = pd.DataFrame([{"KPI": k, "ถ้าไม่มี AI": f(a), "ใช้ AI": f(b), "ต่าง": f"{b - a:+,.0f}",
                   "ดีขึ้น": "✅" if ((b < a) if lower else (b > a)) else "❌"} for k, a, b, lower in rows])
st.dataframe(t, hide_index=True, width="stretch")
st.info("บิลรวมลดลงเล็กน้อย เพราะ AI ชาร์จ–จ่ายมากขึ้นจนค่าพลังงานสูงขึ้น "
        "ส่วนที่คุ้มชัดคือพีคลดลงและไฟวูบไม่ทำให้ไลน์หยุด")
st.caption("สมมติฐานในข้อมูลจำลอง (ต้องยืนยันด้วยค่าจริง): Demand Charge 132.93 บาท/kW/เดือน · "
           "โอกาสซับไฟวูบ กฎเดิม 30% / AI 95% · หยุดไลน์ 5–30 นาที เสียหาย 1,000 บาท/นาที")