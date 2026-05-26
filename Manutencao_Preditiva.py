import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def gerar_relatorio_pcm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Cálculos para a "tradução"
    total = len(y_test)
    recall = tp / (tp + fn)
    
    print("--- RELATÓRIO DE IMPACTO OPERACIONAL (PCM) ---")
    print(f"Total de equipamentos monitorados: {total}")
    print(f"Falhas críticas detectadas: {tp}")
    print(f"Alarmes falsos gerados: {fp}")
    print(f"Eficiência de Detecção (Recall): {recall:.2%}")
    
    if recall > 0.95:
        print("Status: Modelo de Alta Confiabilidade.")
    else:
        print("Status: Modelo requer ajuste nos parâmetros de sensibilidade.")

# ---------------------------------------- Importação e tratamento da base ----------------------------------------
df = pd.read_csv("ai4i2020.csv")

df_limpo = df.drop(columns=['UDI', 'Product ID', 'Type'])

# ---------------------------------------- Treinando e testando o modelo ----------------------------------------

X = df_limpo.drop(columns=['Machine failure'])
y = df_limpo['Machine failure']

X_train_raw, X_test_raw = train_test_split(df_limpo.drop(columns=['Machine failure']), test_size=0.2, random_state=42)

scaler = StandardScaler()

train_data, test_data, y_train, y_test = train_test_split(
    df_limpo.drop(columns=['Machine failure']), 
    df_limpo['Machine failure'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df_limpo['Machine failure']
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

model.fit(X_train_scaled, y_train)

importances = model.feature_importances_
feature_names = train_data.columns

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=importances, y=feature_names, palette='viridis')
ax.bar_label(ax.containers[0], fmt='%.3f', padding=3)

plt.title("Importância das Variáveis no Modelo")
plt.xlabel("Peso da Variável")
plt.ylabel("Sensor / Indicador")
plt.xlim(0, max(importances) * 1.2) 
plt.tight_layout()
plt.savefig("sensor_falhas.png", dpi=300)
plt.close()

y_pred = model.predict(X_test_scaled)

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão: Modelo de Manutenção")
plt.ylabel("Valor Real")
plt.xlabel("Previsão")
plt.savefig("matriz_confusao.png", dpi=300)
plt.close() 

gerar_relatorio_pcm(y_test, y_pred)

# ---------------------------------------- Gerando arquivo excel + gráfico ----------------------------------------

df_resultado = test_data.copy()
df_resultado['Previsao'] = y_pred

writer = pd.ExcelWriter("Falhas_Iminentes.xlsx", engine='xlsxwriter')

manutencao_necessaria = df_resultado[df_resultado['Previsao'] == 1]

manutencao_necessaria.to_excel(writer, sheet_name='Lista_Visitas', index=False)

workbook  = writer.book
worksheet = workbook.add_worksheet('Analise_Grafica')
writer.sheets['Analise_Grafica'] = worksheet

worksheet.insert_image('A2', 'sensor_falhas.png')

writer.close()
