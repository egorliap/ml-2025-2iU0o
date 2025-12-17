import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
data = np.array([
    [1, 2, -1, 4, 10],
    [3, -3, -3, 12, -15],
    [2, 1, -2, 4, 5],
    [5, 1, -5, 10, 5],
    [2, 3, -3, 5, 12],
    [4, 0, -3, 16, 2],
])

print("Исходные данные:")
print(data)

# Шаг 1: Стандартизация
standardized_data = (data - data.mean(axis=0)) / data.std(axis=0)
print("\nСтандартизированные данные:")
print(standardized_data)

# Шаг 2: Матрица ковариации
covariance_matrix = np.cov(standardized_data.T)
print("\nМатрица ковариации:")
print(covariance_matrix)

# Шаг 3: Собственные значения и векторы
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("\nСобственные значения:")
print(eigenvalues)
print("\nСобственные векторы:")
print(eigenvectors)

# Шаг 4: Сортировка по важности
order_of_importance = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[order_of_importance]
sorted_eigenvectors = eigenvectors[:, order_of_importance]
print("\nОтсортированные собственные значения:")
print(sorted_eigenvalues)
print("\nОтсортированные собственные векторы:")
print(sorted_eigenvectors)

# Шаг 5: Объясненная дисперсия
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
print("\nОбъясненная дисперсия по компонентам:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")

# Шаг 6: Преобразование данных
k = 2
reduced_data = np.matmul(standardized_data, sorted_eigenvectors[:, :k])
print(f"\nДанные после PCA (форма: {reduced_data.shape}):")
print(reduced_data)

# Шаг 7: Общая объясненная дисперсия
total_explained_variance = sum(explained_variance[:k])
print(f"\nОбщая объясненная дисперсия для {k} компонент: {total_explained_variance:.3f}")

# График 1: Объясненная дисперсия по компонентам
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
components = [f'PC{i+1}' for i in range(len(explained_variance))]
plt.bar(components, explained_variance * 100, color='skyblue')
plt.xlabel('Главные компоненты')
plt.ylabel('Объясненная дисперсия (%)')
plt.title('Объясненная дисперсия по компонентам')
plt.grid(True, alpha=0.3)

# График 2: Данные после PCA
plt.subplot(1, 2, 2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='red', s=100)

# Подписи точек
labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
for i, label in enumerate(labels):
    plt.text(reduced_data[i, 0] + 0.1, reduced_data[i, 1] + 0.1, label,
             fontsize=12, ha='center', va='center')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Данные после PCA')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()