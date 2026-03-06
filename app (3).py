# -*- coding: utf-8 -*-
"""
ESGFP Interactive Scoring — Streamlit App v4
=============================================
Charts: altair only (ships with streamlit — zero extra installs).
NO matplotlib. NO plotly.
"""

import io, math, random, re, copy, json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Any
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ESGFP Scoring Tool", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

alt.themes.enable("default")

st.markdown("""<style>
.stApp{background:#f7f8fa}
section[data-testid="stSidebar"]{background:#edf0f5;border-right:1px solid #d1d9e6}
h1,h2,h3{color:#1a2332}
.stTabs [data-baseweb="tab"]{font-weight:600}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ALTAIR CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def alt_heatmap(df_long, x_col, y_col, color_col, title, w=600, h=350):
    """df_long must have x_col, y_col, color_col columns."""
    base = alt.Chart(df_long).mark_rect().encode(
        x=alt.X(f"{x_col}:N", title=None, sort=None),
        y=alt.Y(f"{y_col}:N", title=None, sort=None),
        color=alt.Color(f"{color_col}:Q", scale=alt.Scale(scheme="tealblues"), title="Value"),
        tooltip=[x_col, y_col, alt.Tooltip(color_col, format=".4f")]
    ).properties(width=w, height=h, title=title)
    text = alt.Chart(df_long).mark_text(fontSize=10).encode(
        x=alt.X(f"{x_col}:N", sort=None), y=alt.Y(f"{y_col}:N", sort=None),
        text=alt.Text(f"{color_col}:Q", format=".3f"),
        color=alt.condition(alt.datum[color_col] > df_long[color_col].median(),
            alt.value("white"), alt.value("black"))
    )
    return base + text

def alt_bar(df_long, x_col, y_col, title, color="#0f8a6e", w=600, h=350, sort="-y"):
    return alt.Chart(df_long).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{x_col}:N", sort=sort, title=None),
        y=alt.Y(f"{y_col}:Q", title="Score"),
        tooltip=[x_col, alt.Tooltip(y_col, format=".4f")]
    ).properties(width=w, height=h, title=title)

def alt_grouped_bar(df_long, x_col, y_col, group_col, title, w=650, h=380):
    return alt.Chart(df_long).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X(f"{group_col}:N", title=None),
        y=alt.Y(f"{y_col}:Q", title="Value"),
        color=alt.Color(f"{x_col}:N", title="Method"),
        xOffset=alt.XOffset(f"{x_col}:N"),
        tooltip=[x_col, group_col, alt.Tooltip(y_col, format=".5f")]
    ).properties(width=w, height=h, title=title)

def alt_stacked_bar(df_long, x_col, y_col, stack_col, title, w=600, h=380):
    return alt.Chart(df_long).mark_bar().encode(
        x=alt.X(f"{x_col}:N", title=None, sort=None),
        y=alt.Y(f"{y_col}:Q", title="Probability", stack="normalize"),
        color=alt.Color(f"{stack_col}:N", title="Rank"),
        tooltip=[x_col, stack_col, alt.Tooltip(y_col, format=".3f")]
    ).properties(width=w, height=h, title=title)

def alt_donut(df, label_col, value_col, title, w=350, h=350):
    return alt.Chart(df).mark_arc(innerRadius=60, outerRadius=120).encode(
        theta=alt.Theta(f"{value_col}:Q"),
        color=alt.Color(f"{label_col}:N", title="Pillar"),
        tooltip=[label_col, alt.Tooltip(value_col, format=".1f")]
    ).properties(width=w, height=h, title=title)

def alt_radar(pillar_scores, title):
    """Radar approximated as a layered line chart (altair has no native polar)."""
    pillars = pillar_scores.index.tolist(); techs = pillar_scores.columns.tolist()
    norm = (pillar_scores / 90.6) * 10.0
    rows = []
    for t in techs:
        for p in pillars:
            rows.append({"Alternative": t, "Pillar": p, "Score": float(norm.loc[p, t])})
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Pillar:N", title=None, sort=pillars),
        y=alt.Y("Score:Q", title="Normalised (0–10)", scale=alt.Scale(domain=[0, 10])),
        color=alt.Color("Alternative:N"),
        tooltip=["Alternative", "Pillar", alt.Tooltip("Score", format=".2f")]
    ).properties(width=600, height=380, title=title)

def alt_scatter_materiality(df, title, size_by_weight=False):
    """Materiality scatter on Likelihood vs Impact with colored background approximation."""
    plot_df = df.copy()
    plot_df["x"] = (plot_df["Likelihood"] - 1) / 4
    plot_df["y"] = 1 + (plot_df["Impact"] - 1) * 9 / 4
    plot_df["RiskScore"] = plot_df["x"] * 0.5 + (plot_df["y"] / 10) * 0.5

    # Background gradient approximation
    bg_rows = []
    for xi in np.linspace(0, 1, 30):
        for yi in np.linspace(1, 10, 30):
            bg_rows.append({"bx": xi, "by": yi, "bz": 0.5 * xi + 0.5 * ((yi - 1) / 9)})
    bg_df = pd.DataFrame(bg_rows)
    bg = alt.Chart(bg_df).mark_rect(opacity=0.7).encode(
        x=alt.X("bx:Q", bin=alt.Bin(step=1/30), title="Probability (Likelihood)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("by:Q", bin=alt.Bin(step=9/30), title="Severity (Impact)", scale=alt.Scale(domain=[1, 10])),
        color=alt.Color("bz:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None)
    )

    sz_val = "Weight:Q" if size_by_weight and "Weight" in plot_df.columns else alt.value(120)
    pts = alt.Chart(plot_df).mark_circle(stroke="white", strokeWidth=2).encode(
        x=alt.X("x:Q", title="Probability (Likelihood)", scale=alt.Scale(domain=[-0.02, 1.02])),
        y=alt.Y("y:Q", title="Severity (Impact)", scale=alt.Scale(domain=[0.5, 10.5])),
        size=alt.Size("Weight:Q", scale=alt.Scale(range=[80, 400]), legend=None) if (size_by_weight and "Weight" in plot_df.columns) else alt.value(120),
        color=alt.Color("Pillar:N", scale=alt.Scale(
            domain=list(PCOL.keys()), range=list(PCOL.values()))),
        tooltip=["Issue", "Pillar", "Likelihood", "Impact",
                 alt.Tooltip("Score:Q", format=".1f")]
    )
    labels = alt.Chart(plot_df).mark_text(dy=-12, fontSize=9, fontWeight="bold").encode(
        x="x:Q", y="y:Q", text="Issue:N"
    )
    return (bg + pts + labels).properties(width=580, height=420, title=title)

def alt_interval(df, title, winner):
    """Horizontal interval chart for weight stability."""
    bars = alt.Chart(df.reset_index()).mark_bar(height=14, color="#10b981", opacity=0.6).encode(
        x=alt.X("min_pct:Q", title="Weight (%)"),
        x2="max_pct:Q",
        y=alt.Y("Pillar:N", sort=None, title=None),
        tooltip=["Pillar", alt.Tooltip("min_pct", format=".1f"), alt.Tooltip("max_pct", format=".1f")]
    )
    dots = alt.Chart(df.reset_index()).mark_point(shape="diamond", size=120, color="#ef4444", filled=True).encode(
        x="baseline_pct:Q", y=alt.Y("Pillar:N", sort=None),
        tooltip=["Pillar", alt.Tooltip("baseline_pct", format=".1f")]
    )
    return (bars + dots).properties(width=550, height=max(200, 30 * len(df)), title=f"{title} · Winner: {winner}")

PCOL = {"Environment": "#10b981", "Social": "#3b82f6", "Governance": "#8b5cf6",
        "Finance": "#f59e0b", "Process": "#ef4444"}


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR MODEL
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class IndicatorDef:
    pillar: str; key_issue: str; indicator: str; unit: str
    formula_desc: str; criteria: str; default_mode: str; higher_is_better: bool

Model = Dict[str, Dict[str, List[IndicatorDef]]]

RAW_INDICATORS_TSV = """Pillar\tKeyIssue\tIndicator\tUnit\tHigherIsBetter
Environment\tCarbon Efficiency\tNet Carbon Avoided Cost\tUSD/metric ton CO2-e\t0
Environment\tCarbon Efficiency\tTotal Carbon Emissions (Scope 1 & 2)\tmetric tons CO2-e\t0
Environment\tCarbon Efficiency\tCarbon Intensity per Unit\tkg CO2/ton product\t0
Environment\tEnergy Efficiency\tSpecific Energy Consumption\tMJ/ton product\t0
Environment\tEnergy Efficiency\tRenewable Energy Utilization\t% of total energy\t1
Environment\tWater Management\tWater Intensity\tm3/ton product\t0
Environment\tWater Management\tWater Stress Impact\tindex score\t0
Environment\tWaste Management\tWaste-to-Product Conversion\t% conversion\t1
Environment\tWaste Management\tHazardous Waste Generation\tkg/ton product\t0
Environment\tWaste Management\tRecyclable Input Material\t% recycled\t1
Environment\tOperational Efficiency\tProcess Downtime\t% downtime\t0
Environment\tOperational Efficiency\tMaterial Conversion Efficiency\t% conversion\t1
Environment\tOperational Efficiency\tUnit Production Cost\tUSD/ton\t0
Environment\tEnergy Optimization\tEnergy Efficiency Improvement\t% improvement\t1
Environment\tPollution Control\tProcess-Related Air Pollutants\tkg/ton product\t0
Environment\tClean Technology\tAdoption of Cleaner Technologies\t% processes\t1
Environment\tSustainable Products\tRevenue from Green Products\t% revenue\t1
Environment\tR&D Investment\tInvestment in Sustainable Technology\t% revenue\t1
Environment\tLifecycle Impact\tProduct Carbon Footprint\tkg CO2-e/unit\t0
Environment\tEnvironmental Sustainability (SGI)\tPM2.5 / Air Pollution Exposure\tug/m3\t0
Social\tOccupational Health & Safety\tInherent Safety Index (ISI)\tindex score\t1
Social\tOccupational Health & Safety\tOHS Score\tscore (0-100)\t1
Social\tOccupational Health & Safety\tFatal Occupational Injuries\tper 100,000 workers\t0
Social\tOccupational Health & Safety\tNon-fatal Occupational Injuries\tper 100,000 workers\t0
Social\tOccupational Health & Safety\tLabor Inspection Rate\tinspections/1000 employees\t1
Social\tEmployment\tUnemployment Rate\t%\t0
Social\tEmployment\tDirect & Indirect Jobs\tnumber\t1
Social\tEmployment\tEmployment Score\tscore (0-100)\t1
Social\tSocial Protection\tPopulation Covered by Social Protection\t% population\t1
Social\tLabor Rights\tCompliance with Labor Rights\tscore (0-100)\t1
Social\tCommunity Impact\tPopulation Exposure Index\tindex\t0
Social\tCommunity Impact\tSocial Sustainability (SGI)\tindex\t1
Governance\tGender Equality\tFemale to Male Labor Force Participation Ratio\tratio\t1
Governance\tGender Equality\tWomen in Management Positions\t%\t1
Governance\tRegulatory Quality\tRegulatory Quality Estimate\tindex\t1
Governance\tRights\tEconomic & Social Rights Score\tindex\t1
Governance\tInnovation\tR&D Expenditure\t% GDP\t1
Governance\tTrade & Logistics\tLead Time to Export\tdays\t0
Governance\tTrade & Logistics\tLead Time to Import\tdays\t0
Governance\tTrade & Logistics\tLogistics Performance Index\t1-5\t1
Governance\tGovernance Burden\tTime Dealing with Regulators\t% management time\t0
Governance\tGender Inclusion\tFirms with Female Top Manager\t% firms\t1
Governance\tGender Inclusion\tFirms with Female Ownership\t% firms\t1
Governance\tFinance Access\tFirms Using Banks for Working Capital\t% firms\t1
Governance\tLabor Market\tFemale Labor Force Participation\t%\t1
Governance\tCybersecurity\tCyber Crisis Management (NCSI)\tindex\t1
Governance\tCybersecurity\tCyber Incident Response Capacity\tindex\t1
Governance\tCybersecurity\tProtection of Personal Data\tindex\t1
Governance\tCybersecurity\tCyber Threat Awareness\tindex\t1
Governance\tDigitalization\tDigital Development Level\tindex\t1
Governance\tAccountability\tConsensus Building (SGI)\tindex\t1
Governance\tAccountability\tHorizontal Accountability\tindex\t1
Governance\tAccountability\tDiagonal Accountability\tindex\t1
Governance\tBusiness Environment\tStarting a Business\tindex\t1
Governance\tBusiness Environment\tDealing with Construction Permits\tindex\t1
Governance\tBusiness Environment\tGetting Electricity\tindex\t1
Governance\tBusiness Environment\tRegistering Property\tindex\t1
Governance\tBusiness Environment\tGetting Credit\tindex\t1
Governance\tBusiness Environment\tProtecting Minority Investors\tindex\t1
Governance\tBusiness Environment\tGINI Index\tindex\t0
Governance\tBusiness Environment\tPaying Taxes\tindex\t1
Governance\tBusiness Environment\tTrading Across Borders\tindex\t1
Governance\tBusiness Environment\tEnforcing Contracts\tindex\t1
Governance\tBusiness Environment\tResolving Insolvency\tindex\t1
Governance\tPolicy Quality\tSustainable Policymaking (SGI)\tindex\t1
Governance\tPolicy Quality\tEconomic Sustainability (SGI)\tindex\t1
Finance\tESG Financing\tESG-Linked Financing\tUSD (million)\t1
Finance\tCost Structure\tTotal Cost\tUSD (million)\t0
Finance\tReturns\tROI\t%\t1
Finance\tRisk\tNPV\tUSD (million)\t1
Finance\tRisk\tIRR\t%\t1
Process\tMaterials\tEquipment Fabrication Material (encoded)\tscore\t1
Process\tMaterials\tCorrosion Rate\tmm/year\t0
Process\tEnergy\tSpecific Energy Consumption\tMJ/kg\t0
Process\tHazard\tChemical Hazard Risk\trisk score\t0
"""

def _dfm(ind,unit,hib):
    t=f"{ind} {unit}".lower()
    if "rank" in t or "index" in t: return "C"
    return "A" if hib else "B"
def _dff(mode):
    if mode=="A": return "Higher-is-better: CS=30+60*norm; single=90"
    if mode=="B": return "Lower-is-better: CS=30+60*norm; single=90"
    return "Index/ranking deciles (0..90)"
def parse_model(tsv):
    df=pd.read_csv(io.StringIO(tsv),sep="\t"); df.columns=df.columns.str.strip()
    for c in ["Pillar","KeyIssue","Indicator","Unit"]: df[c]=df[c].astype(str).str.strip()
    df["HigherIsBetter"]=df["HigherIsBetter"].astype(int); model={}
    for _,r in df.iterrows():
        hib=bool(int(r["HigherIsBetter"])); dm=_dfm(str(r["Indicator"]),str(r["Unit"]),hib)
        idef=IndicatorDef(str(r["Pillar"]),str(r["KeyIssue"]),str(r["Indicator"]),str(r["Unit"]),_dff(dm),"Criterion",dm,hib)
        model.setdefault(idef.pillar,{}).setdefault(idef.key_issue,[]).append(idef)
    for p in model:
        for ki in model[p]: model[p][ki]=sorted(model[p][ki],key=lambda x:x.indicator.lower())
    return dict(sorted(model.items(),key=lambda kv:kv[0].lower()))


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIALITY
# ═══════════════════════════════════════════════════════════════════════════════
RISKS_CATALOG=OrderedDict({"Financial":["Financial/Market Risk","Liquidity Risk","Cost of Capital Risk"],
    "Technology":["Data & Tech Risk","Cyber-Physical Risk","AI/Model Risk"],
    "Regulatory":["Regulatory Risk","Compliance Risk","Sanctions Risk"],
    "Environmental":["Physical Risk","Transition Risk","Pollution Risk"],
    "Operational":["Supply Chain Risk","Health & Safety Risk","Quality Risk"]})
ALL_RISKS=[r for cat in RISKS_CATALOG.values() for r in cat]
MA_PILLARS=OrderedDict({"Environment":{"color":"#10b981","issues":["Carbon Efficiency","Energy Efficiency","Water Management","Waste Management","Pollution Control"]},
    "Social":{"color":"#3b82f6","issues":["Occupational Health & Safety","Employment","Labor Rights","Community Impact"]},
    "Governance":{"color":"#8b5cf6","issues":["Regulatory Quality","Cybersecurity","Accountability","Policy Quality"]},
    "Finance":{"color":"#f59e0b","issues":["ESG Financing","Cost Structure","Returns"]},
    "Process":{"color":"#ef4444","issues":["Materials","Energy Usage","Hazard Control"]}})

def _rl(s):
    if s<=4: return "Low"
    if s<=9: return "Medium"
    if s<=16: return "High"
    return "Very High"

class MAEngine:
    def __init__(self):
        self.issues=[]; self.risk_data={}; self.stake_data={}
        for pn,pd_ in MA_PILLARS.items():
            for nm in pd_["issues"]: self.add_issue(nm,pn,pd_["color"])
    def add_issue(self,name,pillar,color):
        if any(n==name for n,_,_ in self.issues): return False
        self.issues.append((name,pillar,color))
        self.risk_data[name]={"likelihood":3,"risks":[]}
        self.stake_data[name]={"likelihood":3,"impact":3,"stake":5,"expert":5}; return True
    def _calc_impact(self,risks):
        mx=max((len(d["risks"]) for d in self.risk_data.values()),default=0)
        if mx==0 or len(risks)==0: return 1
        return min(round(1+(len(risks)/mx)*4),5)
    def risk_analysis_df(self):
        rows=[]
        for nm,p,c in self.issues:
            d=self.risk_data[nm]; imp=self._calc_impact(d["risks"]); sc=d["likelihood"]*imp
            rows.append({"Issue":nm,"Pillar":p,"Color":c,"Likelihood":d["likelihood"],"Impact":imp,"Score":sc,"Level":_rl(sc)})
        return pd.DataFrame(rows).sort_values("Score",ascending=False).reset_index(drop=True)
    def stakeholder_df(self):
        rows=[]
        for nm,p,c in self.issues:
            d=self.stake_data[nm]; w=round((d["stake"]+d["expert"])/20,2); base=d["likelihood"]*d["impact"]; sc=round(base*w,2)
            rows.append({"Issue":nm,"Pillar":p,"Color":c,"Likelihood":d["likelihood"],"Impact":d["impact"],"Weight":w,"BaseScore":base,"Score":sc,"Level":_rl(base)})
        return pd.DataFrame(rows).sort_values("Score",ascending=False).reset_index(drop=True)
    def average_df(self):
        ra=self.risk_analysis_df(); st_=self.stakeholder_df(); rows=[]
        for nm,p,c in self.issues:
            rr=ra[ra["Issue"]==nm].iloc[0]; sr=st_[st_["Issue"]==nm].iloc[0]
            al=round((rr["Likelihood"]+sr["Likelihood"])/2,2); ai=round((rr["Impact"]+sr["Impact"])/2,2); asc=round(al*ai,2)
            rows.append({"Issue":nm,"Pillar":p,"Color":c,"RA_Score":rr["Score"],"ST_Score":sr["Score"],"Avg_Likelihood":al,"Avg_Impact":ai,"Avg_Score":asc,"Level":_rl(asc)})
        df=pd.DataFrame(rows); df["Normalized"]=0.0
        for pp in df["Pillar"].unique():
            m=df["Pillar"]==pp; s=df.loc[m,"Avg_Score"].sum()
            if s>0: df.loc[m,"Normalized"]=round(df.loc[m,"Avg_Score"]/s,4)
        return df.sort_values("Avg_Score",ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EXACT SCORING FORMULAS (from CLI)
# ═══════════════════════════════════════════════════════════════════════════════
INDICATOR_SCORE_SCALE=90.0; GM_MIN=0.5; GM_MAX=0.6
PILLAR_SCORE_THEORETICAL_MAX=INDICATOR_SCORE_SCALE+GM_MAX; MIN_DISPLAY_SCORE=1.0; OUTPUT_SCALE=10.0
EXPOSURE_MAP={"High":9.0,"Moderate":7.5,"Low-to-Moderate":5.0,"Low":2.5}

def compute_is(cur,vmin,vmax,hib,n_alts):
    if n_alts<=1: return 90.0
    dn=vmax-vmin
    if abs(dn)<1e-12: return 90.0
    norm=(cur-vmin)/dn if hib else (vmax-cur)/dn
    return float(30.0+60.0*float(np.clip(norm,0,1)))
def compute_rank_cs(cur,best,worst,lb):
    if math.isclose(best,worst): return 0.0
    lo,hi=min(best,worst),max(best,worst)
    if cur<lo or cur>hi: return 0.0
    pos=(cur-lo)/(hi-lo); perf=(1-pos) if lb else pos
    return float(min(int(math.floor(float(np.clip(perf,0,1))*10)),9)*10)
def compute_gm(ge): return 0.1*float(np.clip(ge/10,0,1))+0.5
def compute_ps(is_s,gm,sign): return float(max(0,is_s+sign*gm))
def compute_final(ps,w,n): return ps*(float(w)/max(1,int(n)))
def _ik(p,ki,ind): return f"{p}:{ki}:{ind}"
def compute_ki(sba,model,sel):
    techs=list(sba.keys()); out={}
    for p in sorted(sel.keys()):
        rows=[]
        for ki in sel[p]:
            inds=model.get(p,{}).get(ki,[])
            if not inds: continue
            row={"KI":ki}
            for t in techs: row[t]=sum(sba[t].get(_ik(p,ki,x.indicator),0) for x in inds)
            rows.append(row)
        if rows: out[p]=pd.DataFrame(rows).set_index("KI")[techs]
    return out
def compute_ps_df(sba,model,sel):
    techs=list(sba.keys()); rows=[]
    for p in sorted(sel.keys()):
        row={"Pillar":p}
        for t in techs:
            tot=0
            for ki in sel[p]:
                for x in model.get(p,{}).get(ki,[]): tot+=sba[t].get(_ik(p,ki,x.indicator),0)
            row[t]=tot
        rows.append(row)
    return pd.DataFrame(rows).set_index("Pillar")[techs]


# ═══════════════════════════════════════════════════════════════════════════════
# AHP/FAHP
# ═══════════════════════════════════════════════════════════════════════════════
RI_T={1:0,2:0,3:.58,4:.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49,11:1.51,12:1.48,13:1.56,14:1.57,15:1.59}
TFN={1:(1,1,1),2:(1,2,3),3:(2,3,4),4:(3,4,5),5:(4,5,6),6:(5,6,7),7:(6,7,8),8:(7,8,9),9:(9,9,9)}
def ns(x): x=max(1/9,min(9,x)); return int(min(range(1,10),key=lambda k:abs(k-x)))
def cm(issues,ratings):
    n=len(issues); A=np.ones((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                ratio=ratings[i]/max(ratings[j],1e-9)
                A[i,j]=float(ns(ratio)) if ratio>=1 else 1.0/float(ns(1.0/ratio))
    for i in range(n):
        for j in range(i+1,n): A[j,i]=1.0/A[i,j]
    return A
def ahp_ra(A): cs=A.sum(axis=0); nm=A/cs; w=nm.mean(axis=1); return w/w.sum()
def ahp_gm(A): n=A.shape[0]; g=np.prod(A,axis=1)**(1.0/n); return g/g.sum()
def ahp_ei(A):
    vals,vecs=np.linalg.eig(A); idx=int(np.argmax(vals.real)); lm=float(vals[idx].real)
    w=np.abs(vecs[:,idx].real); return w/w.sum(),lm
def ahp_c(n,lm):
    ci=0 if n<=2 else (lm-n)/(n-1); ri=RI_T.get(n,RI_T[max(RI_T)]); cr=0 if ri==0 else ci/ri; return lm,ci,ri,cr
def c2t(v):
    if v>=1: return TFN[ns(v)]
    inv=1/v; t=TFN[ns(inv)]; return (1/t[2],1/t[1],1/t[0])
def bf(A):
    n=A.shape[0]; return [[(1,1,1) if i==j else c2t(float(A[i,j])) for j in range(n)] for i in range(n)]
def fb(F):
    n=len(F); g=[]
    for i in range(n):
        lp=mp=up=1
        for j in range(n): l,m,u=F[i][j]; lp*=l; mp*=m; up*=u
        g.append((lp**(1/n),mp**(1/n),up**(1/n)))
    sl=sum(x[0] for x in g); sm=sum(x[1] for x in g); su=sum(x[2] for x in g)
    inv=(1/su,1/sm,1/sl); fw=[(l*inv[0],m*inv[1],u*inv[2]) for l,m,u in g]
    d=np.array([(l+m+u)/3 for l,m,u in fw]); return d/d.sum()
def _pg(a,b):
    l1,m1,u1=a; l2,m2,u2=b
    if m1>=m2: return 1.0
    if l2>=u1: return 0.0
    dn=(m1-u1)-(m2-l2); return float(max(0,min(1,(l2-u1)/dn))) if abs(dn)>1e-12 else 0.0
def fc(F):
    n=len(F); rs=[]
    for i in range(n):
        s=(0,0,0)
        for j in range(n): l,m,u=F[i][j]; s=(s[0]+l,s[1]+m,s[2]+u)
        rs.append(s)
    t=(0,0,0)
    for r in rs: t=(t[0]+r[0],t[1]+r[1],t[2]+r[2])
    inv=(1/t[2],1/t[1],1/t[0]); S=[(r[0]*inv[0],r[1]*inv[1],r[2]*inv[2]) for r in rs]
    d=np.zeros(n)
    for i in range(n):
        vals=[_pg(S[i],S[k]) for k in range(n) if k!=i]; d[i]=min(vals) if vals else 1
    return d/d.sum() if d.sum()>0 else np.ones(n)/n
def sp_corr(m2w):
    methods=list(m2w.keys()); m=len(methods); C=np.eye(m)
    for i in range(m):
        for j in range(i+1,m):
            rx=pd.Series(m2w[methods[i]]).rank(method="average").values; ry=pd.Series(m2w[methods[j]]).rank(method="average").values
            cx,cy=rx-rx.mean(),ry-ry.mean(); dn=float(np.sqrt((cx**2).sum())*np.sqrt((cy**2).sum()))
            c=float((cx*cy).sum()/dn) if dn>0 else 0; C[i,j]=c; C[j,i]=c
    return pd.DataFrame(C,index=methods,columns=methods)


# ═══════════════════════════════════════════════════════════════════════════════
# MCDA + VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def _wv(pl,wp): w=np.array([float(wp.get(p,0)) for p in pl]); s=w.sum(); return w/(s if s else 1)
def _dm(ps): return ps.to_numpy().T,list(ps.columns),list(ps.index)
def mcda_w(ps,wp): A,al,cr=_dm(ps); w=_wv(cr,wp); return pd.Series(A@w,index=al).sort_values(ascending=False)
def mcda_t(ps,wp):
    A,al,cr=_dm(ps); w=_wv(cr,wp); nm=np.linalg.norm(A,axis=0); nm[nm==0]=1; R=A/nm; V=R*w
    ib,iw=V.max(0),V.min(0); db=np.linalg.norm(V-ib,axis=1); dw=np.linalg.norm(V-iw,axis=1)
    return pd.Series(dw/(db+dw+1e-12),index=al).sort_values(ascending=False)
def mcda_v(ps,wp,v=0.5):
    A,al,cr=_dm(ps); w=_wv(cr,wp); fs,fm=A.max(0),A.min(0); dn=fs-fm; dn[dn==0]=1; gap=(fs-A)/dn
    S=(gap*w).sum(1); R=(gap*w).max(1); QS=(S-S.min())/(S.max()-S.min()+1e-12); QR=(R-R.min())/(R.max()-R.min()+1e-12)
    return pd.Series(1-(v*QS+(1-v)*QR),index=al).sort_values(ascending=False)
def mcda_e(ps,wp):
    A,al,cr=_dm(ps); w=_wv(cr,wp); avg=A.mean(0); avg[avg==0]=1
    PDA=np.maximum(0,(A-avg)/avg); NDA=np.maximum(0,(avg-A)/avg); SP=PDA@w; SN=NDA@w
    return pd.Series(((SP/(SP.max()+1e-12))+(1-SN/(SN.max()+1e-12)))/2,index=al).sort_values(ascending=False)
MF={"WEIGHTED":mcda_w,"TOPSIS":mcda_t,"VIKOR":mcda_v,"EDAS":mcda_e}
def _sc(s,method):
    if s.empty: return s
    m=method.upper()
    if m=="WEIGHTED": return ((s/PILLAR_SCORE_THEORETICAL_MAX)*OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)
    if m in {"TOPSIS","VIKOR","EDAS"}: return (s*OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)
    lo,hi=float(s.min()),float(s.max())
    if math.isclose(hi,lo): return pd.Series([5]*len(s),index=s.index)
    return (((s-lo)/(hi-lo))*OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)

def _dir(a): samples=np.random.gamma(shape=a,scale=1); s=samples.sum(); return samples/s if s>0 else np.ones_like(a)/len(a)
def dea(ps,samp=2000):
    al=list(ps.columns); Y=ps.to_numpy().T; m=Y.shape[0]; fh=np.zeros(m,dtype=int); ph=np.ones(m)
    for i in range(m):
        ot=[k for k in range(m) if k!=i]
        if not ot: continue
        Yo=Y[ot,:]; br=1.0
        for _ in range(samp):
            la=_dir(np.ones(len(ot))); r=float(np.min((la@Yo)/np.maximum(Y[i,:],1e-9)))
            if r>=1: fh[i]+=1
            if r>br: br=r
        ph[i]=max(br,1)
    return pd.DataFrame({"Alt":al,"FrontierProb":fh/float(samp),"EffOut":1/np.maximum(ph,1e-12)}).set_index("Alt")
def mc(ps,methods,sims=1000,alpha=1.0,sigma=0.03):
    methods=[m.upper() for m in methods]
    if "WEIGHTED" not in methods: methods=["WEIGHTED"]+methods
    al=ps.columns.tolist(); pl=ps.index.tolist(); m=len(al); n=len(pl)
    bc={mt:{a:0 for a in al} for mt in methods}; rc={mt:np.zeros((m,m),dtype=int) for mt in methods}
    A0=ps.to_numpy().T; av=np.full(n,float(alpha))
    for _ in range(sims):
        w=_dir(av); A=np.clip(A0+np.random.normal(0,sigma,A0.shape),0,None)
        dfA=pd.DataFrame(A.T,index=pl,columns=al); wp={p:w[i]*100 for i,p in enumerate(pl)}
        pm={mt:MF[mt](dfA,wp) for mt in methods if mt in MF}
        for mt,s in pm.items():
            bc[mt][s.sort_values(ascending=False).index[0]]+=1
            ranks=s.rank(ascending=False,method="average").loc[al].values
            for ia,rk in enumerate(ranks.astype(int)): rc[mt][ia,max(0,min(m-1,rk-1))]+=1
    rpb=[]
    for mt in methods:
        for a in al: rpb.append({"Method":mt,"Alt":a,"PB":bc[mt][a]/sims})
    Pb=pd.DataFrame(rpb).pivot(index="Alt",columns="Method",values="PB").fillna(0)
    RD={mt:pd.DataFrame(rc[mt]/sims,index=al,columns=[f"Rank {k}" for k in range(1,m+1)]) for mt in methods}
    return Pb,RD
def smaa(ps,sims=2000,alpha=1.0):
    al=ps.columns.tolist(); m=len(al); n=len(ps.index); bct=np.zeros(m); A=ps.to_numpy().T; av=np.full(n,float(alpha))
    for _ in range(sims):
        w=_dir(av); bct[int(np.argmax(A@w))]+=1
    return pd.Series(bct/sims,index=al).sort_values(ascending=False)
def wstab(ps,bw):
    al=ps.columns.tolist(); pl=ps.index.tolist(); w0=_wv(pl,bw); A=ps.to_numpy().T
    sc0=A@w0; wi=int(np.argmax(sc0)); winner=al[wi]; intervals=[]
    for j,pil in enumerate(pl):
        wj0=float(w0[j]); do=1-wj0
        if abs(do)<1e-12: intervals.append((pil,wj0*100,100,100)); continue
        tmin,tmax=0.0,1.0
        for bi,b in enumerate(al):
            if b==winner: continue
            delta=A[wi,:]-A[bi,:]; Aj=float(delta[j]); B=float(np.dot(np.delete(w0,j),np.delete(delta,j))/do)
            c_=Aj-B; d_=B
            if abs(c_)<1e-12:
                if d_<-1e-12: tmin,tmax=1,0; break
                continue
            bound=-d_/c_
            if c_>0: tmin=max(tmin,bound)
            else: tmax=min(tmax,bound)
        tmin=float(np.clip(tmin,0,1)); tmax=float(np.clip(tmax,0,1))
        if tmin>tmax: tmin,tmax=float("nan"),float("nan")
        intervals.append((pil,wj0*100,tmin*100,tmax*100))
    return winner,pd.DataFrame(intervals,columns=["Pillar","baseline_pct","min_pct","max_pct"]).set_index("Pillar")


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k,v in {"model":None,"ma_engine":None,"selected_pillars":{},"ahp_weights":{},"ahp_method":"FAHP Buckley",
    "issue_weights":{},"alt_labels":[],"scores_by_alt":{},"pillar_scores":None,"key_issue_scores":{},
    "scenario_results":[],"scenario_weights":[],"scenario_methods":[],"alt_kind":"technology"}.items():
    if k not in st.session_state: st.session_state[k]=v
if st.session_state["model"] is None: st.session_state["model"]=parse_model(RAW_INDICATORS_TSV)
if st.session_state["ma_engine"] is None: st.session_state["ma_engine"]=MAEngine()

pages=["⓪ Framework Editor","① Materiality","② Setup","③ AHP/FAHP","④ Scoring",
       "⑤ Summary","⑥ Scenarios","⑦ Validation","⑧ Export"]
with st.sidebar:
    st.markdown("## 📊 ESGFP Scoring"); page=st.radio("Navigation",pages,index=0)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════
if page==pages[0]:
    st.header("⓪ Framework Editor"); model=st.session_state["model"]
    rows=[]
    for p in sorted(model.keys()):
        for ki in sorted(model[p].keys()):
            for x in model[p][ki]: rows.append({"Pillar":p,"Key Issue":ki,"Indicator":x.indicator,"Unit":x.unit,"Mode":x.default_mode,"Dir":"↑" if x.higher_is_better else "↓"})
    if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=350)
    c1,c2=st.columns(2)
    with c1:
        st.subheader("➕ Add"); np_=st.text_input("Pillar",key="fp"); nki=st.text_input("Key Issue",key="fk"); nind=st.text_input("Indicator",key="fi"); nu=st.text_input("Unit",value="index",key="fu")
        nm_=st.selectbox("Mode",["A","B","C"],key="fm")
        if st.button("Add",type="primary") and np_ and nki and nind:
            model.setdefault(np_,{}).setdefault(nki,[]).append(IndicatorDef(np_,nki,nind,nu,_dff(nm_),"Criterion",nm_,nm_=="A")); st.success("Added"); st.rerun()
    with c2:
        st.subheader("➖ Remove"); rp=st.selectbox("Pillar",["—"]+sorted(model.keys()),key="rp")
        if rp!="—":
            rki=st.selectbox("Key Issue",["—"]+sorted(model.get(rp,{}).keys()),key="rk")
            if rki!="—":
                ri_=st.selectbox("Indicator",["(whole KI)"]+[x.indicator for x in model.get(rp,{}).get(rki,[])],key="ri")
                if st.button("Remove"):
                    if ri_=="(whole KI)": del model[rp][rki]
                    else: model[rp][rki]=[x for x in model[rp][rki] if x.indicator!=ri_]
                    if rp in model and rki in model.get(rp,{}) and not model[rp][rki]: del model[rp][rki]
                    if rp in model and not model[rp]: del model[rp]
                    st.rerun()

elif page==pages[1]:
    st.header("① Materiality"); ma=st.session_state["ma_engine"]
    tabs=st.tabs(["Inputs","Risk Analysis","Stakeholder","Combined"])
    with tabs[0]:
        for pn in MA_PILLARS:
            p_iss=[(n,p,c) for n,p,c in ma.issues if p==pn]
            if not p_iss: continue
            with st.expander(f"**{pn}**"):
                for nm,_,_ in p_iss:
                    c1,c2=st.columns(2)
                    with c1: ma.risk_data[nm]["likelihood"]=st.slider(f"RA L: {nm}",1,5,ma.risk_data[nm]["likelihood"],key=f"ra_{nm}"); ma.risk_data[nm]["risks"]=st.multiselect(f"Risks: {nm}",ALL_RISKS,default=ma.risk_data[nm]["risks"],key=f"rr_{nm}")
                    with c2: ma.stake_data[nm]["likelihood"]=st.slider(f"ST L: {nm}",1,5,ma.stake_data[nm]["likelihood"],key=f"sl_{nm}"); ma.stake_data[nm]["impact"]=st.slider(f"ST I: {nm}",1,5,ma.stake_data[nm]["impact"],key=f"si_{nm}"); ma.stake_data[nm]["stake"]=st.slider(f"Stake: {nm}",0,10,ma.stake_data[nm]["stake"],key=f"ss_{nm}"); ma.stake_data[nm]["expert"]=st.slider(f"Expert: {nm}",0,10,ma.stake_data[nm]["expert"],key=f"se_{nm}")
    with tabs[1]:
        ra=ma.risk_analysis_df(); st.altair_chart(alt_scatter_materiality(ra,"Risk Analysis"),use_container_width=True); st.dataframe(ra[["Issue","Likelihood","Impact","Score","Level"]],hide_index=True)
    with tabs[2]:
        st_=ma.stakeholder_df(); st.altair_chart(alt_scatter_materiality(st_,"Stakeholder",True),use_container_width=True); st.dataframe(st_[["Issue","Likelihood","Impact","Weight","Score","Level"]],hide_index=True)
    with tabs[3]:
        avg=ma.average_df(); renamed=avg.rename(columns={"Avg_Likelihood":"Likelihood","Avg_Impact":"Impact","Avg_Score":"Score"})
        st.altair_chart(alt_scatter_materiality(renamed,"Combined"),use_container_width=True); st.dataframe(avg[["Issue","Pillar","RA_Score","ST_Score","Avg_Score","Normalized","Level"]].round(3),hide_index=True)

elif page==pages[2]:
    st.header("② Setup"); model=st.session_state["model"]; st.session_state["alt_kind"]=st.radio("Type",["Technology","Process Design"]).lower(); sel={}
    for p in sorted(model.keys()):
        with st.expander(f"📌 {p}",expanded=True): sel[p]=st.multiselect(p,sorted(model[p].keys()),default=sorted(model[p].keys()),key=f"ki_{p}")
    st.session_state["selected_pillars"]=sel; st.success(f"✅ {sum(len(v) for v in sel.values())} key issues")

elif page==pages[3]:
    st.header("③ AHP/FAHP"); model=st.session_state["model"]; sel=st.session_state.get("selected_pillars",{})
    if not sel: st.warning("Setup first."); st.stop()
    ar={}
    for pillar in sorted(sel.keys()):
        issues=sel[pillar]
        if not issues: continue
        with st.expander(f"📌 {pillar}",expanded=True):
            if len(issues)==1: ar[pillar]=pd.DataFrame({"Key Issue":issues,"AHP RowAvg":[1],"AHP Eigen":[1],"AHP GeoMean":[1],"FAHP Buckley":[1],"FAHP Chang":[1]}); st.info("w=1"); continue
            cols=st.columns(min(len(issues),4)); ratings=[]
            for i,ki in enumerate(issues):
                with cols[i%len(cols)]: ratings.append(st.slider(ki,1,9,1,key=f"a_{pillar}_{ki}"))
            A=cm(issues,ratings); wra=ahp_ra(A); wgm=ahp_gm(A)
            try: wei,lm=ahp_ei(A); lv,ci,ri,cr=ahp_c(len(issues),lm)
            except: wei=wra; lv=ci=ri=cr=0
            F=bf(A); wb_=fb(F); wc_=fc(F)
            df_w=pd.DataFrame({"Key Issue":issues,"AHP RowAvg":wra,"AHP Eigen":wei,"AHP GeoMean":wgm,"FAHP Buckley":wb_,"FAHP Chang":wc_}).sort_values("FAHP Buckley",ascending=False)
            ar[pillar]=df_w
            if len(issues)>=3: mc_=st.columns(4); mc_[0].metric("λ",f"{lv:.3f}"); mc_[1].metric("CI",f"{ci:.4f}"); mc_[2].metric("RI",f"{ri:.2f}"); mc_[3].metric("CR",f"{cr:.4f}")
            st.dataframe(df_w.style.format({c:"{:.5f}" for c in df_w.columns if c!="Key Issue"}),use_container_width=True,hide_index=True)
            # Heatmap
            hm_rows=[]
            for i,r_ in enumerate(issues):
                for j,c_ in enumerate(issues): hm_rows.append({"Row":r_,"Col":c_,"Value":float(A[i,j])})
            st.altair_chart(alt_heatmap(pd.DataFrame(hm_rows),"Col","Row","Value",f"{pillar} — Pairwise"),use_container_width=True)
            # Weights bar
            wlong=df_w.melt(id_vars="Key Issue",var_name="Method",value_name="Weight")
            st.altair_chart(alt_grouped_bar(wlong,"Method","Weight","Key Issue",f"{pillar} — Weights"),use_container_width=True)
    st.markdown("---"); chosen=st.selectbox("Method",["FAHP Buckley","AHP RowAvg","AHP Eigen","AHP GeoMean","FAHP Chang"])
    st.session_state["ahp_method"]=chosen; st.session_state["ahp_weights"]=ar
    iw={}
    for p,dfw in ar.items():
        d={str(r["Key Issue"]):float(r[chosen]) for _,r in dfw.iterrows()}; s=sum(d.values())
        if s>0: d={k:v/s for k,v in d.items()}
        iw[p]=d
    st.session_state["issue_weights"]=iw; st.success(f"✅ {chosen}")

elif page==pages[4]:
    st.header("④ Scoring"); model=st.session_state["model"]; sel=st.session_state.get("selected_pillars",{}); iw=st.session_state.get("issue_weights",{})
    if not iw: st.warning("AHP first."); st.stop()
    ts="technology" if "tech" in st.session_state["alt_kind"] else "process design"
    n_alt=st.number_input(f"# {ts}s",1,20,2); labels=[]
    nc=st.columns(min(int(n_alt),5))
    for i in range(int(n_alt)):
        with nc[i%len(nc)]: labels.append(st.text_input(f"{i+1}",f"{ts.title()} {i+1}",key=f"an_{i}"))
    st.session_state["alt_labels"]=labels; sba={l:{} for l in labels}
    for p in sorted(sel.keys()):
        st.markdown(f"### {p}")
        for ki in sel[p]:
            inds=model.get(p,{}).get(ki,[]); w=iw.get(p,{}).get(ki,0); nc_=max(1,len(inds))
            if not inds: continue
            with st.expander(f"**{ki}** (w={w:.4f})",expanded=False):
                for x in inds:
                    st.markdown(f"`{x.indicator}` [{x.unit}] {'↑' if x.higher_is_better else '↓'}")
                    ic=st.columns(3)
                    with ic[0]: mode=st.selectbox("Mode",["A","B","C"],index=["A","B","C"].index(x.default_mode),key=f"m_{p}_{ki}_{x.indicator}")
                    with ic[1]: ge=EXPOSURE_MAP[st.selectbox("GE",list(EXPOSURE_MAP.keys()),key=f"ge_{p}_{ki}_{x.indicator}")]
                    with ic[2]: gsi=1 if st.selectbox("±",["+","-"],key=f"gs_{p}_{ki}_{x.indicator}")=="+" else -1
                    gml=compute_gm(ge)
                    if mode in ("A","B"):
                        hib=(mode=="A"); vals={}; vc=st.columns(len(labels))
                        for il,lab in enumerate(labels):
                            with vc[il]: vals[lab]=st.number_input(lab,value=0.0,format="%.4f",key=f"v_{p}_{ki}_{x.indicator}_{lab}")
                        vmn,vmx=min(vals.values()),max(vals.values())
                        for lab in labels: sba[lab][_ik(p,ki,x.indicator)]=round(compute_final(compute_ps(compute_is(vals[lab],vmn,vmx,hib,len(labels)),gml,gsi),w,nc_),6)
                    else:
                        rc=st.columns(3)
                        with rc[0]: best=st.number_input("Best",value=1.0,key=f"b_{p}_{ki}_{x.indicator}")
                        with rc[1]: worst=st.number_input("Worst",value=200.0,key=f"w_{p}_{ki}_{x.indicator}")
                        with rc[2]: lb=st.checkbox("Lower=better",True,key=f"lb_{p}_{ki}_{x.indicator}")
                        vc=st.columns(len(labels))
                        for il,lab in enumerate(labels):
                            with vc[il]: rv=st.number_input(lab,value=1.0,key=f"rk_{p}_{ki}_{x.indicator}_{lab}"); sba[lab][_ik(p,ki,x.indicator)]=round(compute_final(compute_ps(compute_rank_cs(rv,best,worst,lb),gml,gsi),w,nc_),6)
    st.session_state["scores_by_alt"]=sba; st.session_state["key_issue_scores"]=compute_ki(sba,model,sel); st.session_state["pillar_scores"]=compute_ps_df(sba,model,sel)
    st.success("✅ Done → Summary")

elif page==pages[5]:
    st.header("⑤ Summary"); ps=st.session_state.get("pillar_scores"); kis=st.session_state.get("key_issue_scores",{})
    if ps is None or ps.empty: st.warning("Score first."); st.stop()
    st.dataframe(ps.round(4),use_container_width=True)
    # Heatmap
    hm=[]
    for p in ps.index:
        for t in ps.columns: hm.append({"Pillar":p,"Alternative":t,"Score":float(ps.loc[p,t])})
    st.altair_chart(alt_heatmap(pd.DataFrame(hm),"Alternative","Pillar","Score","Pillar Scores"),use_container_width=True)
    st.altair_chart(alt_radar(ps,"Radar (0–10 normalised)"),use_container_width=True)
    ov=ps.sum(axis=0).sort_values(ascending=False)
    st.altair_chart(alt_bar(pd.DataFrame({"Alt":ov.index,"Score":ov.values}),"Alt","Score","Overall Summary"),use_container_width=True)
    for p_,df_ in kis.items():
        if df_.empty: continue
        with st.expander(f"📌 {p_}"): st.dataframe(df_.round(5),use_container_width=True)

elif page==pages[6]:
    st.header("⑥ Scenarios"); ps=st.session_state.get("pillar_scores")
    if ps is None or ps.empty: st.warning("Score first."); st.stop()
    pillars=ps.index.tolist(); n_sc=st.number_input("Scenarios",1,10,1); asr=[]; asw=[]; asm=[]
    for si in range(int(n_sc)):
        with st.expander(f"**Scenario {si+1}**",expanded=True):
            wc=st.columns(len(pillars)); wpct={}
            for j,p_ in enumerate(pillars):
                with wc[j]: wpct[p_]=st.number_input(f"{p_}%",0.0,100.0,100.0/len(pillars),step=1.0,key=f"sw_{si}_{p_}")
            tw=sum(wpct.values())
            if abs(tw-100)>0.5: st.error(f"Total={tw:.1f}%")
            msel=st.multiselect("Methods",["WEIGHTED","TOPSIS","VIKOR","EDAS"],default=["WEIGHTED","TOPSIS"],key=f"mt_{si}")
            if "WEIGHTED" not in msel: msel=["WEIGHTED"]+msel
            if abs(tw-100)<=0.5:
                pm={mt:MF[mt](ps,wpct) for mt in msel if mt in MF}; sdf=pd.DataFrame(pm)
                scaled={c:_sc(sdf[c],c) for c in sdf.columns}; ss=pd.DataFrame(scaled,index=sdf.index)
                st.dataframe(ss.round(4),use_container_width=True); asr.append(ss); asw.append(wpct); asm.append(msel)
                wlong=ss.reset_index().melt(id_vars="index",var_name="Method",value_name="Score").rename(columns={"index":"Alt"})
                st.altair_chart(alt_grouped_bar(wlong,"Method","Score","Alt",f"Scenario {si+1}"),use_container_width=True)
                st.altair_chart(alt_donut(pd.DataFrame({"Pillar":list(wpct.keys()),"Weight":list(wpct.values())}),"Pillar","Weight",f"Weights {si+1}"),use_container_width=True)
    st.session_state["scenario_results"]=asr; st.session_state["scenario_weights"]=asw; st.session_state["scenario_methods"]=asm

elif page==pages[7]:
    st.header("⑦ Validation"); ps=st.session_state.get("pillar_scores"); sw=st.session_state.get("scenario_weights",[]); sm_=st.session_state.get("scenario_methods",[])
    if ps is None or ps.empty or not sw: st.warning("Scenarios first."); st.stop()
    si=st.selectbox("Scenario",[f"S{i+1}" for i in range(len(sw))]); idx=int(si[1:])-1; wpct=sw[idx]; methods=sm_[idx]
    ds=st.number_input("DEA samp",500,10000,2000,500); mcs=st.number_input("MC sims",100,10000,1000,500)
    if st.button("▶ Run",type="primary",use_container_width=True):
        with st.spinner("DEA..."): ds_=dea(ps,int(ds))
        st.subheader("DEA"); st.dataframe(ds_.round(4)); st.altair_chart(alt_bar(ds_.reset_index().rename(columns={"Alt":"Alt_"}),"Alt_","EffOut","DEA Efficiency"),use_container_width=True)
        st.markdown("---")
        with st.spinner("MC..."): Pb,RD=mc(ps,methods,int(mcs))
        st.subheader("Monte-Carlo")
        if "WEIGHTED" in Pb.columns:
            pb_w=Pb["WEIGHTED"].sort_values(ascending=False)
            st.altair_chart(alt_bar(pd.DataFrame({"Alt":pb_w.index,"PBest":pb_w.values}),"Alt","PBest","P(Best) WEIGHTED"),use_container_width=True)
        for mt in RD:
            rd_=RD[mt]; rl=rd_.reset_index().melt(id_vars=rd_.index.name or "index",var_name="Rank",value_name="Prob")
            if rd_.index.name is None: rl=rl.rename(columns={"index":"Alt"})
            st.altair_chart(alt_stacked_bar(rl,rd_.index.name or "Alt","Prob","Rank",f"Rankogram {mt}"),use_container_width=True)
        st.markdown("---")
        with st.spinner("SMAA..."): pb_s=smaa(ps)
        st.subheader("SMAA"); st.altair_chart(alt_bar(pd.DataFrame({"Alt":pb_s.index,"PBest":pb_s.values}),"Alt","PBest","SMAA P(Best)"),use_container_width=True)
        st.markdown("---")
        with st.spinner("Stability..."): winner,idf=wstab(ps,wpct)
        st.subheader("Weight Stability"); st.altair_chart(alt_interval(idf,"Critical Weight",winner),use_container_width=True)

elif page==pages[8]:
    st.header("⑧ Summary & Export"); ps=st.session_state.get("pillar_scores"); sba=st.session_state.get("scores_by_alt",{}); sr=st.session_state.get("scenario_results",[])
    if ps is None or ps.empty: st.warning("Complete workflow."); st.stop()
    ov=ps.sum(axis=0).sort_values(ascending=False); st.success(f"🏆 **{ov.index[0]}** — {ov.values[0]:.4f}")
    st.dataframe(ps.round(4),use_container_width=True)
    st.altair_chart(alt_bar(pd.DataFrame({"Alt":ov.index,"Score":ov.values}),"Alt","Score","Final Ranking"),use_container_width=True)
    st.markdown("---"); st.subheader("📥 Downloads")
    c1,c2,c3=st.columns(3)
    with c1: st.download_button("Pillar Scores CSV",ps.round(6).to_csv().encode(),"pillar_scores.csv")
    with c2: st.download_button("Summary CSV",pd.DataFrame({"Alt":ov.index,"Score":ov.values,"Rank":range(1,len(ov)+1)}).to_csv(index=False).encode(),"summary.csv")
    with c3:
        if sba:
            rows=[]
            for alt,scores in sba.items():
                for k,v in scores.items():
                    parts=k.split(":",2)
                    if len(parts)==3: rows.append({"Alt":alt,"Pillar":parts[0],"KI":parts[1],"Indicator":parts[2],"Score":v})
            if rows: st.download_button("Detail CSV",pd.DataFrame(rows).to_csv(index=False).encode(),"detail.csv")
    st.download_button("📦 Full JSON",json.dumps({"timestamp":datetime.now().isoformat(),"scores":ps.round(6).to_dict(),"ranking":ov.round(6).to_dict()},indent=2,default=str).encode(),"assessment.json","application/json")
