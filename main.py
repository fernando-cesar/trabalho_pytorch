import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("default of credit card clients.xls", skiprows=1)
print(data.head())

data_clean = data.drop(columns=["ID", "SEX"])
print(data_clean.head())
total = data_clean.isnull().sum()
percent = (data_clean.isnull().sum() / data_clean.isnull().count() * 100)
print(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose())

# busca por outliers
outliers = {}
for i in range(data_clean.shape[1]):
    min_t = data_clean[data_clean.columns[i]].mean() - (3 * data_clean[data_clean.columns[i]].std())
    max_t = data_clean[data_clean.columns[i]].mean() + (3 * data_clean[data_clean.columns[i]].std())
    count = 0
    for j in data_clean[data_clean.columns[i]]:
        if j < min_t or j > max_t:
            count += 1
    percentage = count/data_clean.shape[0]
    outliers[data_clean.columns[i]] = "%.3f" % percentage
print(outliers)

# desequilíbrio de classes
target = data_clean["default payment next month"]
sim = target[target == 1].count()
nao = target[target == 0].count()
print("\nsim %: " + str(sim/len(target)*100) + " - nao %: " + str(nao/len(target) * 100))

fig, ax = plt.subplots(figsize=(10, 5))
plt.bar("sim", sim)
plt.bar("nao", nao)
ax.set_yticks([sim, nao])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# reamostragem da classe subrepresentada
data_sim = data_clean[data_clean["default payment next month"] == 1]
data_nao = data_clean[data_clean["default payment next month"] == 0]
over_sampling = data_sim.sample(nao, replace=True, random_state=0)
data_resampled = pd.concat([data_nao, over_sampling], axis=0)

data_resampled = data_resampled.reset_index(drop=True)
X = data_resampled.drop(columns=["default payment next month"])
y = data_resampled ["default payment next month"]

X = (X - X.min())/(X.max() - X.min())
print(X.head())

# salvando dados normalizados em arquivo csv
final_data = pd.concat([X, y], axis=1)
final_data.to_csv("dccc_prepared.csv", index=False)
