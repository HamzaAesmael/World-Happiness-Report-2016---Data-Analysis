# 1. Data Preparation & Cleaning
# Import necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import plotly.figure_factory as ff
import missingno as msno
import plotly.io as pio
import os

# Create a directory for saving visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

print("=" * 60)
print("WORLD HAPPINESS REPORT 2016 - ANALYSIS")
print("=" * 60)

# Load the dataset
df = pd.read_csv('2016.csv')
print(" First 5 rows :")
print(df.head(5))
print(" The data describtion : ")
print(df.describe())

# Initial inspection 
print(" Dataset Shape : ", df.shape)
print(" Missing Values : ")
print(df.isnull().sum())
# Vis the missing values in the columns 
plt.figure(figsize=(10, 8))
ax = msno.matrix(df,
                 labels=True,  # Show column names on x-axis
                 sparkline=False, # Hide the sparkline on the right
                 figsize=(12, 8),
                 color=(0.1, 0.5, 0.75),  # Nice blue color
                 fontsize=12)
plt.title('Missing Values Analysis: World Happiness Report 2016\n', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('\nColumns', fontsize=12, fontweight='bold')
plt.ylabel('Record Index', fontsize=12, fontweight='bold')
plt.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.3)
plt.tight_layout() 
plt.show()
# CONVERT STRING COLUMNS TO NUMERIC FIRST
numeric_columns = ['Health (Life Expectancy)', 'Lower Confidence Interval', 
                  'Upper Confidence Interval', 'Economy (GDP per Capita)',
                  'Family', 'Freedom', 'Trust (Government Corruption)', 
                  'Generosity', 'Dystopia Residual', 'Happiness Score']
for col in numeric_columns  :
    if col in df.columns : 
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Converted {col} to numeric")


# Handling missing values
df['Health (Life Expectancy)'].fillna(df['Health (Life Expectancy)'].median(), inplace=True)
df['Lower Confidence Interval'].fillna(df['Lower Confidence Interval'].median(), inplace=True)
df['Upper Confidence Interval'].fillna(df['Upper Confidence Interval'].median(), inplace=True)
df['Economy (GDP per Capita)'].fillna(df['Economy (GDP per Capita)'].median(), inplace=True) 
df['Freedom'].fillna(df['Freedom'].median(), inplace=True)  
# Confirm no more missing values
print("\nMissing Values after cleaning:")
print(df.isnull().sum())

# 2. Exploratory Data Analysis & Visualizations
# Get top 10 countries by Happiness Rank (since rank 1 is happiest)
top10 = df.nsmallest(10,'Happiness Rank')
plt.figure(figsize=(14,8))
x = np.arange(len(top10))
width = 0.35
plt.bar(x-width/2,top10['Economy (GDP per Capita)'],width, label='GDP per Capita', color='darkblue', alpha=0.8)
plt.bar(x+width/2,top10['Health (Life Expectancy)'],width, label='Health (Life Expectancy)', color='lightseagreen', alpha=0.8)
plt.xlabel('Country', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Top 10 Happiest Countries: GDP vs Health Life Expectancy', fontsize=14, fontweight='bold')
plt.xticks(x, top10['Country'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/top10_gdp_health.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved top10_gdp_health.png")

# Visualization Correlation Heatmap (Seaborn)
plt.figure(figsize=(14,8))
corr_cols = ['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 
             'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Happiness Score']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Happiness Factors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved correlation_heatmap.png")

## Visualization 4: Scatter Plot - GDP vs Happiness by Region
plt.figure(figsize=(14,8))
regions = df['Region'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
for i, region in enumerate(regions):
    region_data = df[df['Region'] == region]
    plt.scatter(region_data['Economy (GDP per Capita)'], 
                region_data['Happiness Score'],
                alpha=0.7, s=60, color=colors[i], label=region)
    
plt.xlabel('GDP per Capita', fontsize=12, fontweight='bold')
plt.ylabel('Happiness Score', fontsize=12, fontweight='bold')
plt.title('Happiness Score vs GDP per Capita by Region', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/gdp_vs_happiness.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved gdp_vs_happiness.png")

# Visualization 5: Pie Chart - Average Happiness by Region
plt.figure(figsize=(12, 8))
region_avg = df.groupby('Region')['Happiness Score'].mean().sort_values(ascending=False)

plt.pie(region_avg.values, labels=region_avg.index, autopct='%1.1f%%', 
        startangle=90, colors=plt.cm.Paired.colors)
plt.title('Average Happiness Score by Region', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('visualizations/happiness_by_region_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved happiness_by_region_pie.png")

# Visualization 6: World Map of GDP per Capita (Plotly HTML)
fig_map = px.choropleth(df,
                     locations="Country",
                     locationmode='country names',
                     color="Economy (GDP per Capita)",
                     hover_name="Country",
                     hover_data=["Health (Life Expectancy)", "Happiness Score"],
                     color_continuous_scale=px.colors.sequential.Plasma,
                     title="World Map: GDP per Capita (2016)")
fig_map.write_html("visualizations/world_gdp_map.html")
print("✓ Saved world_gdp_map.html")

# 3. Create a Dashboard Summary Image
print("\n" + "=" * 40)
print("CREATING DASHBOARD SUMMARY")
print("=" * 40)

# Create a 2x2 dashboard grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Top 10 Countries
x = np.arange(len(top10))
ax1.bar(x - 0.2, top10['Economy (GDP per Capita)'], 0.4, label='GDP', alpha=0.8)
ax1.bar(x + 0.2, top10['Health (Life Expectancy)'], 0.4, label='Health', alpha=0.8)
ax1.set_title('Top 10: GDP & Health Life Expectancy', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(top10['Country'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Correlation Heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax2.set_xticks(range(len(corr_cols)))
ax2.set_yticks(range(len(corr_cols)))
ax2.set_xticklabels([x[:15] + '...' if len(x) > 15 else x for x in corr_cols], rotation=45, ha='right')
ax2.set_yticklabels([x[:15] + '...' if len(x) > 15 else x for x in corr_cols])
ax2.set_title('Correlation Matrix', fontweight='bold')
plt.colorbar(im, ax=ax2)

# Plot 3: Scatter plot
for i, region in enumerate(regions[:6]):  # Show first 6 regions for clarity
    region_data = df[df['Region'] == region]
    ax3.scatter(region_data['Economy (GDP per Capita)'], 
                region_data['Happiness Score'],
                alpha=0.7, s=40, label=region[:20] + '...' if len(region) > 20 else region)
ax3.set_xlabel('GDP per Capita')
ax3.set_ylabel('Happiness Score')
ax3.set_title('Happiness vs GDP by Region', fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)

# Plot 4: Pie chart
ax4.pie(region_avg.values, labels=region_avg.index, autopct='%1.1f%%', 
        startangle=90, colors=plt.cm.Set3.colors)
ax4.set_title('Avg Happiness by Region', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/happiness_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved happiness_dashboard.png")

# 4. Final Summary
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print("All visualizations have been saved in the 'visualizations' folder:")
print("1. missing_values.png - Missing data analysis")
print("2. top10_gdp_health.png - Top 10 countries comparison")
print("3. correlation_heatmap.png - Factor correlations")
print("4. gdp_vs_happiness.png - Regional analysis")
print("5. happiness_by_region_pie.png - Regional happiness distribution")
print("6. world_gdp_map.html - Interactive world map (open in browser)")
print("7. happiness_dashboard.png - Summary dashboard")

print("\nKey Insights:")
print(f"- Dataset contains {df.shape[0]} countries and {df.shape[1]} features")
print(f"- Highest Happiness Score: {df['Happiness Score'].max():.2f}")
print(f"- Strongest correlation with Happiness: Economy ({corr_matrix.loc['Economy (GDP per Capita)', 'Happiness Score']:.2f})")
print("- Western Europe and ANZ regions show highest average happiness")

print("\nCheck the 'visualizations' folder for all generated charts!")