import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('IVV ETF Stock Price History.csv')
df = df.dropna()

def convert_volume(vol_str):
    if pd.isna(vol_str):
        return np.nan
    vol_str = str(vol_str).strip()
    if vol_str.endswith('M'):
        return float(vol_str[:-1]) * 1_000_000
    elif vol_str.endswith('K'):
        return float(vol_str[:-1]) * 1_000
    elif vol_str.endswith('B'):
        return float(vol_str[:-1]) * 1_000_000_000
    else:
        return float(vol_str.replace(',', ''))
df['Vol.'] = df['Vol.'].apply(convert_volume)

def convert_change_percent(change_str):
    if pd.isna(change_str):
        return np.nan
    change_str = str(change_str).strip().replace('%', '')
    return float(change_str) / 100

df['Change %'] = df['Change %'].apply(convert_change_percent)
df['Close'] = (df['Open'] * (1 + df['Change %'])).round(2)
df['Gamma'] = np.where(df['Open'].diff() > 0, 1, -1)
df.loc[0, 'Gamma'] = -1
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features_to_correlate = [col for col in numeric_cols if col != 'Gamma']
correlations = df[features_to_correlate].corrwith(df['Gamma'])

print("Final Dataset:")
print(df.head(10))
print("Correlations with Γ(Gamma)")
print(correlations.sort_values(ascending=False))

df['Cumulative_Gamma'] = df['Gamma'].cumsum()
df['Date'] = pd.to_datetime(df['Date'])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(df['Date'], df['Open'], linewidth=1.5, color='blue')
ax1.set_title('Behavior of Openvalues', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Open ($)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2.plot(df['Date'], df['Cumulative_Gamma'], linewidth=1.5, color='red')
ax2.set_title('Γ Cumulative Movement', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Cummulative Γ', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('step3_plots.png', dpi=300, bbox_inches='tight')