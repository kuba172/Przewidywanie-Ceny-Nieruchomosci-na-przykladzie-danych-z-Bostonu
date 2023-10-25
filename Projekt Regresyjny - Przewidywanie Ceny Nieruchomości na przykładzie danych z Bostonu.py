#!/usr/bin/env python
# coding: utf-8

# **Projekt Regresyjny: Przewidywanie Ceny Nieruchomości na przykładzie danych z Bostonu**
# 
# W tym projekcie zamierzam wykorzystać zbiór danych Boston Housing Dataset, który zawiera informacje o różnych atrybutach nieruchomości w Bostonie. Celem analizy będzie stworzenie modelu regresji, który przewiduje ceny nieruchomości na podstawie tych atrybutów. Po podziale danych na zbiór treningowy i testowy, zastosować zamierzam model regresji liniowej i dokonać oceny jego wydajności, uzyskując satysfakcjonujące wyniki na podstawie miar Mean Squared Error (MSE) oraz R-squared.

# **Krok 1. Załadowanie zbioru danych do projektu i biblioteki do analizy danych w Pythonie - Pandas**
# 
# Pierwszym krokiem jest wczytanie popularnej biblioteki do analizy danych - Pandas, a także próbki danych, które dostępne są w ramach biblioteki sklearn, a konkretniej sklearn.datasets.

# In[23]:


from sklearn.datasets import load_boston
import pandas as pd


# **Krok 2. Wczytanie danych**
# 
# W drugim kroku, za pomocą biblioteki Pandas chciałbym uzyskać wgląd w załączone dane, i uzyskać wgląd w pierwsze 10 wierszy.

# In[24]:


data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target #to chcemy przewidywać
print(df.head(10))


# **Krok 3. Uzyskanie informacji o danych.**
# 
# Następnie chciałbym uzyskać informacje o zbiorze danych, tj. liczbie wierszy, kolumnach i typach danych.

# In[25]:


print(df.info())


# **Krok 4. Podstawowe statystyki.**
# 
# Kolejno sprawdzam podstawowe statystyki dla danych, takie jak średnia, kwantyle, odchylenie standardowe czy wartość minimalna i maksymalna.

# In[26]:


print(df.describe())


# **Krok 5. Wczytanie bibliotek do wizualizacji danych - matplotlib i seaborn.**
# 
# W piątym kroku projektu przeprowadzam wizualną analizę danych, skupiając się na zrozumieniu wzorców i zależności między różnymi cechami. 

# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns


# **Krok 6. Wykres korelacji**
# 
# Wykorzystując wykresy korelacji, można zbadać wzajemne relacje między atrybutami, co pozwoli zidentyfikować istotne zmienne w kontekście przewidywania cen nieruchomości. 

# In[28]:


correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz Korelacji')
plt.show()


# Macierz korelacji to tabelaryczne zestawienie, które pokazuje współzależności pomiędzy różnymi zmiennymi w zbiorze danych. W przypadku analizy regresyjnej, macierz korelacji pozwala zrozumieć, jak zmienne są ze sobą powiązane.
# 
# Wartości w macierzy korelacji mieszczą się w zakresie od -1 do 1:
# 
# - 1 oznacza idealną dodatnią korelację - im jedna zmienna rośnie, tym druga rośnie w proporcji.
# - -1 oznacza idealną ujemną korelację - im jedna zmienna rośnie, tym druga maleje w proporcji.
# - 0 oznacza brak liniowej zależności między zmiennymi.

# **Krok 7. Wykresy rozrzutu dla kilku istotnych cech**
# 
# Za pomocą wykresów rozrzutu zamierzam, zweryfikować potencjalne trendy i związki między poszczególnymi zmiennymi a ceną nieruchomości.

# In[29]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='RM', y='PRICE', data=df)
plt.title('Liczba pokoi vs. Cena')
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='LSTAT', y='PRICE', data=df)
plt.title('Procent mieszkańców o niskim statusie społecznym vs. Cena')
plt.show()


# **Krok 8. Podział na zbiór treningowy i testowy.**
# 
# Dzielę dane na zbiór treningowy - takim na którym model się będzie szkolił, i na testowy - taki, na którym model będzie się sprawdzał w propircji 8:2.

# In[30]:


from sklearn.model_selection import train_test_split

X = df.drop('PRICE', axis=1)
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# **Krok 9. Wybór modelu regresji.**
# 
# Tak jak było od początku założone w tym przypadku bardzo dobrze powinna sprawdzić się regresja liniowa.

# In[31]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# **Krok 10. Ocena modelu.**
# 
# Gdy mamy już model należało by sprawdzić za pomocą odpowiednich metryk jak nasz model sobie radzi. Te metryki to błąd średnio - kwadratowy i R-kwadrat.

# In[32]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# Dobrym pomysłem byłoby też zwizualizowanie regresji.

# In[33]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(np.arange(0, 51), np.arange(0, 51), color='red', linestyle='--') 
plt.xlabel("Cena Rzeczywista")
plt.ylabel("Cena Przewidywana")
plt.title("Regresja - Przewidywane vs. Rzeczywiste Ceny Nieruchomości")
plt.show()


# **Krok 11. Wnioski**
# 
# Ten projekt zawiera kompleksową analizę danych z Boston Housing Dataset, w tym przygotowanie danych, zastosowanie modelu regresji, a także ocenę wydajności modelu. Wyniki zostały zwizualizowane, a dodatkowo dokonano oceny za pomocą miar Mean Squared Error (MSE) oraz R-squared.

# In[ ]:




