import numpy as np

# 1. Знайти в датасеті таргет. Завантажити датасет без таргета
# - таргет = col 5
dataset = np.loadtxt(fname="iris.data", delimiter=",", usecols=(0, 1, 2, 3))
print(f"Dataset:\n{dataset[:3]}...")

# 2. Перетворити колонки, що залишились в 2D масив (або впевнитись, що це уже 2D масив)
if dataset.ndim == 2:
    print("\nIt is a two dimensional array\n")
else:
    print("\nIt is not a two dimensional array\n")

# 3. Порахувати mean, median, standard deviation для 1-ї колонки
print(f"First Column:")
print(f"- mean: {np.mean(dataset[:, 0])}")
print(f"- median: {np.median(dataset[:, 0])}")
print(f"- standard deviation: {np.std(dataset[:, 0])}")

# 4. Вставити 20 значень np.nan на випадкові позиції в масиві
# (при використанні звичайного рандому можуть накластись позиції,
# тому знайти рішення, яке гарантує 20 унікальних позицій)

random_row_indexes = np.random.choice(a=dataset.shape[0] - 1, size=20, replace=False)
random_column_indexes = np.random.choice(a=dataset.shape[1] - 1, size=20, replace=True)

print(f"\nRandom Indexes:\n{random_row_indexes}\n{random_column_indexes}\n")
print(f"Random Index Values:\n{dataset[random_row_indexes, random_column_indexes]}")

dataset[random_row_indexes, random_column_indexes] = np.nan

print(
    f"\nReplaces Random Index Values:\n{dataset[random_row_indexes, random_column_indexes]}"
)


# 5. Знайти позиції вставлених значень np.nan в 1-й колонці
rows = np.where(np.isnan(dataset[:, 0]))[0]
print(f"\nIndexes of NaN values in the first column:\n{rows}")

# 6. Замінити всі значення np.nan на 0
dataset = np.nan_to_num(dataset, nan=0)
print(f"\nReplaced NaN values: \n{dataset[random_row_indexes, random_column_indexes]}")


# 7. Відфільтрувати массив за умовою: значення в 3-й колонці > 1.5
# та значения в 1-й колонці < 5.0 (зберегти у іншу змінну)
filtered_dataset = dataset[(dataset[:, 0] < 5.0) & (dataset[:, 2] > 1.5)]
print(f"\nFiltered dataset:\n{filtered_dataset}")

# 8. Розбити масив по вертикалі на 2 рівні частини (не використовувати
# абсолютні числа, мають бути два массиви по 4 колонки)
first_half, second_half = np.split(dataset, 2)
print(f"\nDivided dataset:\n{first_half}\n\n{second_half}")

# 9. Відсортувати обидва массиви по 1-й колонці: 1-й за збільшенням, 2-й за зменшенням
sorted_first_half = first_half[np.argsort(first_half[:, 0], descending=False)]
sorted_second_half = second_half[np.argsort(second_half[:, 0], descending=True)]
print(f"\nSorted subarrays:\n{sorted_first_half}\n\n{sorted_second_half}")

# 10. Зібрати обидва массиви в одне ціле
concatenated_dataset = np.concat((sorted_first_half, sorted_second_half))
print(f"\nConcatenated dataset:\n{concatenated_dataset}")

# 11. Порахувати всі унікальні значення в массиві та вивести їх разом із кількістю
values, counts = np.unique(dataset, return_counts=True)
print(f"\nUnique values:\n{values}\n\nUnique value count:\n{counts}")

# 12. Знайти найбільш часто повторюване значення в массиві
most_frequent_value = values[np.argmax(counts)]
print(f"\nThe most frequent value: {most_frequent_value}\n")


# 13. Написати функцію, яка б множила всі значення в колонці, які менше середнього
# значения в цій колонці, на 2, і ділила інші значення на 4.
def column_modifier_factory(column):
    mean = np.mean(column)

    def column_modifier(value):
        if value < mean:
            return value * 2
        else:
            return value / 4

    return column_modifier


# 14. Застосувати отриману функцію до 3-ї колонки. Функція має прийняти лише одну
# колонку і повернути колонку із зміненими значеннями
column_modifier_npfn = np.vectorize(column_modifier_factory(dataset[:, 2]))
modified_third_column = column_modifier_npfn(dataset[:, 2])
print(f"\nThird column after modifier function:\n{modified_third_column}")
