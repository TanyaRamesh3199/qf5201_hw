import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import calendar as _cal
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CALENDAR UTILITIES
# ============================================================================

def easter_sunday(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

CONFIRMED_HOLIDAYS = {
    2026: [date(2026,1,1), date(2026,4,3), date(2026,4,6), date(2026,5,1), date(2026,12,25), date(2026,12,26)],
    2027: [date(2027,1,1), date(2027,3,26), date(2027,3,29), date(2027,5,1), date(2027,12,25), date(2027,12,26)],
    2028: [date(2028,1,1), date(2028,4,14), date(2028,4,17), date(2028,5,1), date(2028,12,25), date(2028,12,26)],
}

def target_holidays(year):
    if year in CONFIRMED_HOLIDAYS:
        return CONFIRMED_HOLIDAYS[year]
    es = easter_sunday(year)
    return [date(year,1,1), es - timedelta(days=2), es + timedelta(days=1),
            date(year,5,1), date(year,12,25), date(year,12,26)]

class TargetCalendar:
    def __init__(self, years_buffer=range(2026, 2096)):
        self._hol = set()
        for y in years_buffer:
            self._hol.update(target_holidays(y))

    def is_business_day(self, d):
        if d.weekday() >= 5: return False
        return d not in self._hol

    def following(self, d):
        while not self.is_business_day(d): d += timedelta(days=1)
        return d

    def preceding(self, d):
        while not self.is_business_day(d): d -= timedelta(days=1)
        return d

    def modified_following(self, d):
        f = self.following(d)
        if f.month != d.month: return self.preceding(d)
        return f

    def add_business_days(self, d, n):
        step = 1 if n >= 0 else -1
        n = abs(n)
        while n > 0:
            d += timedelta(days=step)
            if self.is_business_day(d): n -= 1
        return d

CAL = TargetCalendar()

def act360(d1, d2): return (d2 - d1).days / 360.0

def thirty_360(d1, d2):
    d1d = min(d1.day, 30)
    d2d = d2.day
    if d1d == 30 and d2d == 31: d2d = 30
    return ((d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2d - d1d)) / 360.0

def last_day_of_month(d):
    last = _cal.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, last)

def add_months(d, n):
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, _cal.monthrange(y, m)[1])
    return date(y, m, day)

def add_period(d, tenor):
    tenor = tenor.strip().upper()
    n = int(tenor[:-1])
    unit = tenor[-1]
    if unit == 'D': return d + timedelta(days=n)
    if unit == 'W': return d + timedelta(weeks=n)
    if unit == 'M': return add_months(d, n)
    if unit == 'Y': return add_months(d, 12 * n)
    raise ValueError(tenor)

def tenor_months(tenor):
    tenor = tenor.strip().upper()
    n = int(tenor[:-1]); unit = tenor[-1]
    if unit == 'W': return None
    if unit == 'M': return n
    if unit == 'Y': return 12 * n
    raise ValueError(tenor)

def schedule_backward(start, end, step_months, eom_rule=True, cal=CAL):
    is_eom = eom_rule and (end == last_day_of_month(end))
    raw = [end]
    cur = end
    while True:
        prev = add_months(cur, -step_months)
        if is_eom: prev = last_day_of_month(prev)
        if prev <= start: break
        raw.append(prev)
        cur = prev
    raw.append(start)
    raw = sorted(set(raw))
    adj = []
    for d in raw:
        if d == start: adj.append(start)
        else: adj.append(cal.modified_following(d))
    return adj

# ============================================================================
# SECTION 2: SMOOTH CURVE INTERPOLATOR
# ============================================================================

class MonotoneConvexCurve:
    """
    PCHIP interpolation on log ( DF ) . Produces C1 - smooth forward curves . 
    To approximate the shape - preserving properties of Montone Convex method , PCHIP is used 
    """
    def __init__(self):
        self.t = []
        self.df = []
        self._interpolator = None

    def add_pillar(self, t, df):
        if t in self.t:
            idx = self.t.index(t)
            self.df[idx] = df
        else:
            self.t.append(t)
            self.df.append(df)
        
        sorted_pairs = sorted(zip(self.t, self.df))
        self.t = [p[0] for p in sorted_pairs]
        self.df = [p[1] for p in sorted_pairs]
        
        if len(self.t) >= 2:
            self._interpolator = PchipInterpolator(self.t, np.log(self.df))

    def df_t(self, t):
        if t <= 0: return 1.0
        if len(self.t) == 1: return self.df[0]
        if self._interpolator is None: raise ValueError("Curve unbuilt.")
        return float(np.exp(self._interpolator(t)))

# ============================================================================
# SECTION 3: BOOTSTRAPPING SYSTEM
# ============================================================================

REF_DATE = date(2026, 6, 11)
SPOT_DATE = CAL.add_business_days(REF_DATE, 2)

estr_on_rate = 0.0193
estr_swaps = [
    ("1W", 0.0211), ("2W", 0.0215), ("3W", 0.0216), ("1M", 0.0217), ("2M", 0.0220),
    ("3M", 0.0222), ("4M", 0.0227), ("5M", 0.0232), ("6M", 0.0235), ("7M", 0.0239),
    ("8M", 0.0242), ("9M", 0.0244), ("10M", 0.0247), ("11M", 0.0249), ("1Y", 0.0251),
    ("15M", 0.0254), ("18M", 0.0255), ("21M", 0.0256), ("2Y", 0.0256), ("3Y", 0.0257),
    ("4Y", 0.0259), ("5Y", 0.0262), ("6Y", 0.0266), ("7Y", 0.0271), ("8Y", 0.0275),
    ("9Y", 0.0280), ("10Y", 0.0285), ("12Y", 0.0294), ("15Y", 0.0304), ("20Y", 0.0311),
    ("25Y", 0.0311), ("30Y", 0.0308), ("40Y", 0.0298), ("50Y", 0.0285), ("60Y", 0.0273)
]

euribor_depo = [
    ("1W", 0.0189), ("1M", 0.0213), ("3M", 0.0240), ("6M", 0.0262), ("12M", 0.0285)
]

euribor_swaps = [
    ("1Y", 0.0279), ("2Y", 0.0282), ("3Y", 0.0282), ("4Y", 0.0283), ("5Y", 0.0286),
    ("6Y", 0.0289), ("7Y", 0.0293), ("8Y", 0.0297), ("9Y", 0.0301), ("10Y", 0.0305),
    ("12Y", 0.0312), ("15Y", 0.0321), ("20Y", 0.0326), ("30Y", 0.0321)
]

def yf(d): return (d - REF_DATE).days / 365.0

# ----- €STR Bootstrapping -----
estr = MonotoneConvexCurve()
on_end = CAL.following(REF_DATE + timedelta(days=1))
tau_on = act360(REF_DATE, on_end)
df_on = 1.0 / (1.0 + estr_on_rate * tau_on)
estr.add_pillar(yf(on_end), df_on)

for tenor, rate in estr_swaps:
    maturity = CAL.modified_following(add_period(SPOT_DATE, tenor))
    months = tenor_months(tenor) if tenor[-1] != 'W' else None
    
    if months is None or months < 12:
        tau = act360(SPOT_DATE, maturity)
        df_spot = estr.df_t(yf(SPOT_DATE))
        df_mat = df_spot / (1.0 + rate * tau)
        estr.add_pillar(yf(maturity), df_mat)
        continue

    accr = schedule_backward(SPOT_DATE, maturity, 12, eom_rule=True)
    pay = [accr[0]] + [CAL.add_business_days(d, 1) for d in accr[1:]]
    taus = [act360(accr[i], accr[i+1]) for i in range(len(accr)-1)]
    df_spot = estr.df_t(yf(SPOT_DATE))

    known_sum = 0.0
    for i in range(len(taus) - 1):
        known_sum += taus[i] * estr.df_t(yf(pay[i+1]))
    tau_last = taus[-1]

    def eq(df_last):
        pv_fix = rate * (known_sum + tau_last * df_last)
        return pv_fix - (df_spot - df_last)

    sol = brentq(eq, 1e-8, 2.0, xtol=1e-14)
    estr.add_pillar(yf(pay[-1]), sol)

# ----- EURIBOR Bootstrapping -----
euri = MonotoneConvexCurve()
euri.add_pillar(yf(SPOT_DATE), 1.0)

for tenor, rate in euribor_depo:
    maturity = CAL.modified_following(add_period(SPOT_DATE, tenor))
    tau = act360(SPOT_DATE, maturity)
    df_mat = 1.0 / (1.0 + rate * tau)
    euri.add_pillar(yf(maturity), df_mat)

for tenor, rate in euribor_swaps:
    maturity = CAL.modified_following(add_period(SPOT_DATE, tenor))
    fix_sched = schedule_backward(SPOT_DATE, maturity, 12, eom_rule=True)
    flt_sched = schedule_backward(SPOT_DATE, maturity, 6, eom_rule=True)

    pv_fix = 0.0
    for i in range(len(fix_sched) - 1):
        tau_f = thirty_360(fix_sched[i], fix_sched[i+1])
        pv_fix += rate * tau_f * estr.df_t(yf(fix_sched[i+1]))

    known_flt = 0.0
    for i in range(len(flt_sched) - 2):
        t1, t2 = flt_sched[i], flt_sched[i+1]
        known_flt += (euri.df_t(yf(t1)) / euri.df_t(yf(t2)) - 1.0) * estr.df_t(yf(t2))

    t1_last, t2_last = flt_sched[-2], flt_sched[-1]
    disc_last = estr.df_t(yf(t2_last))
    df_s_last = euri.df_t(yf(t1_last))

    def eq(df_last):
        pv_flt = known_flt + (df_s_last / df_last - 1.0) * disc_last
        return pv_fix - pv_flt

    sol = brentq(eq, 1e-8, 2.0, xtol=1e-14)
    euri.add_pillar(yf(t2_last), sol)

# ============================================================================
# SECTION 4: PLOTTING INTERFACES
# ============================================================================

def zero_rate(curve, t):
    df = curve.df_t(t)
    return -np.log(df) / t if t > 0 else 0.0

def estr_on_forward(t):
    dt = 1/365.0
    return (estr.df_t(t) / estr.df_t(t+dt) - 1.0) / dt

def euribor_6m_forward(t):
    dt = 0.5
    return (euri.df_t(t) / euri.df_t(t+dt) - 1.0) / dt

grid = np.linspace(0.01, 5.0, 300)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Subplot 1: Zero Rate Term Curves
axes[0].plot(grid, [zero_rate(estr, t)*100 for t in grid], label='€STR Zero Curve', color='royalblue', lw=2.5)
axes[0].plot(grid, [zero_rate(euri, t)*100 for t in grid], label='6M EURIBOR Zero Curve', color='darkorange', lw=2.5)
axes[0].set_title('Bootstrapped Continuously Compounded Zero Curves', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Maturity Timeline (Years)', fontsize=11)
axes[0].set_ylabel('Zero Rate (%)', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

#Subplot 2: Forward Projections
axes[1].plot(grid, [estr_on_forward(t)*100 for t in grid], label='€STR Instantaneous Forward', color='blue', lw=2)
axes[1].plot(grid, [euribor_6m_forward(t)*100 for t in grid], label='6M EURIBOR Forward', color='orange', lw=2)
axes[1].set_title('EUR Market Forward Rate Projections', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Maturity Timeline (Years)', fontsize=11)
axes[1].set_ylabel('Forward Rate (%)', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()
plt.tight_layout()

#Subplot 3: Discounting Curve
axes[2].plot(grid, [estr.df_t(t) for t in grid], label='€STR Discounting Curve (OIS)', color='forestgreen', lw=2.5)
axes[2].plot(grid, [euri.df_t(t) for t in grid], label='6M EURIBOR Pseudo-DF Curve', color='crimson', lw=2, linestyle='--')
axes[2].set_title('Discount Factor Term Structure', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Maturity Timeline (Years)', fontsize=11)
axes[2].set_ylabel('Discount Factor', fontsize=11)
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].legend()
plt.tight_layout()
plt.show()
