# coding: utf-8

# Começando com os imports
import csv
import matplotlib.pyplot as plt

# Vamos ler os dados como uma lista
print("Lendo o documento...")
with open("chicago.csv", "r") as file_read:
    reader = csv.reader(file_read)
    data_list = list(reader)
print("Ok!")

# Vamos verificar quantas linhas nós temos
print("Número de linhas:")
print(len(data_list))

# Imprimindo a primeira linha de data_list para verificar se funcionou.
print("Linha 0: ")
print(data_list[0])
# É o cabeçalho dos dados, para que possamos identificar as colunas.

# Imprimindo a segunda linha de data_list, ela deveria conter alguns dados
print("Linha 1: ")
print(data_list[1])

input("Aperte Enter para continuar...")

# TAREFA 1
# TODO: Imprima as primeiras 20 linhas usando um loop para identificar os dados.
print("\n\nTAREFA 1: Imprimindo as primeiras 20 amostras")
amostra = 21  # Considerando cabeçalho
for each in range(amostra):
    print(data_list[each])
# Vamos mudar o data_list para remover o cabeçalho dele.
data_list = data_list[1:]

# Nós podemos acessar as features pelo índice
# Por exemplo: sample[6] para imprimir gênero, ou sample[-2]

input("Aperte Enter para continuar...")

# TAREFA 2
# TODO: Imprima o `gênero` das primeiras 20 linhas

print("\nTAREFA 2: Imprimindo o gênero das primeiras 20 amostras")
for each in range(amostra):
    if data_list[each][6] != '':
        print(data_list[each][6])
    else:
        print('Indisponível')
# Ótimo! Nós podemos pegar as linhas(samples) iterando com um for, e as colunas(features) por índices.
# Mas ainda é difícil pegar uma coluna em uma lista. Exemplo: Lista com todos os gêneros

input("Aperte Enter para continuar...")

# TAREFA 3
# TODO: Crie uma função para adicionar as colunas(features) de uma lista em outra lista, na mesma ordem


def column_to_list(data, index):
    """
    Função que cria as features.
    Argumentos:
        data_list: Recebe uma lista de todo o arquivo .csv.
        index: Posição da feature que queremos buscar.
    Retorna:
        Uma lista com as features desejadas.
    """
    column_list = []
    for each in range(len(data)):
        column_list.append(data[each][index])
    # Dica: Você pode usar um for para iterar sobre as amostras, pegar a feature pelo seu índice, e dar append para uma lista
    return column_list


# Vamos checar com os gêneros se isso está funcionando (apenas para os primeiros 20)
print("\nTAREFA 3: Imprimindo a lista de gêneros das primeiras 20 amostras")
print(column_to_list(data_list, -2)[:20])

# ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
assert type(column_to_list(data_list, -2)
            ) is list, "TAREFA 3: Tipo incorreto retornado. Deveria ser uma lista."
assert len(column_to_list(data_list, -2)
           ) == 1551505, "TAREFA 3: Tamanho incorreto retornado."
assert column_to_list(data_list, -2)[0] == "" and column_to_list(
    data_list, -2)[1] == "Male", "TAREFA 3: A lista não coincide."
# -----------------------------------------------------

input("Aperte Enter para continuar...")

# Agora sabemos como acessar as features, vamos contar quantos Male (Masculinos) e Female (Femininos) o dataset tem
# TAREFA 4
# TODO: Conte cada gênero. Você não deveria usar uma função para isso.
male = 0
female = 0
for each in range(len(data_list)):
    if data_list[each][6] == 'Male':
        male += 1
    elif data_list[each][6] == 'Female':
        female += 1

# Verificando o resultado
print("\nTAREFA 4: Imprimindo quantos masculinos e femininos nós encontramos")
print("Masculinos: ", male, "\nFemininos: ", female)

# ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
assert male == 935854 and female == 298784, "TAREFA 4: A conta não bate."
# -----------------------------------------------------

input("Aperte Enter para continuar...")

# Por que nós não criamos uma função para isso?
# TAREFA 5
# TODO: Crie uma função para contar os gêneros. Retorne uma lista.
# Isso deveria retornar uma lista com [count_male, count_female] (exemplo: [10, 15] significa 10 Masculinos, 15 Femininos)


def count_gender(data_list, index=6):
    """
    Função que conta os gêneros
    Argumentos:
        data_list: Recebe uma lista de todo o arquivo .csv.
        index: Posição da feature que queremos buscar. Default é 6.
    Retorna:
        Uma lista onde o primeiro elemento é a qtd de male e o segundo elemento é a qtd de female.
    """
    male = 0
    female = 0
    for each in range(len(data_list)):
        if data_list[each][index] == 'Male':
            male += 1
        elif data_list[each][index] == 'Female':
            female += 1
    return [male, female]


print("\nTAREFA 5: Imprimindo o resultado de count_gender")
print(count_gender(data_list))

# ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
assert type(count_gender(data_list)
            ) is list, "TAREFA 5: Tipo incorreto retornado. Deveria retornar uma lista."
assert len(count_gender(data_list)
           ) == 2, "TAREFA 5: Tamanho incorreto retornado."
assert count_gender(data_list)[0] == 935854 and count_gender(data_list)[
    1] == 298784, "TAREFA 5: Resultado incorreto no retorno!"
# -----------------------------------------------------

input("Aperte Enter para continuar...")

# Agora que nós podemos contar os usuários, qual gênero é mais prevalente?
# TAREFA 6
# TODO: Crie uma função que pegue o gênero mais popular, e retorne este gênero como uma string.
# Esperamos ver "Male", "Female", ou "Equal" como resposta.


def most_popular_gender(data_list):
    """
    Função que retorna o gênero mais popular.
    Argumentos:
        data_list: Recebe uma lista de todo o arquivo .csv.
    Retorna:
        Uma string com Male, Female ou Equal como valor.
    """
    answer = ""
    gender_list = count_gender(data_list)
    if gender_list[0] > gender_list[1]:
        answer = 'Male'
    elif gender_list[0] < gender_list[1]:
        answer = 'Female'
    else:
        answer = 'Equal'
    return answer


print("\nTAREFA 6: Qual é o gênero mais popular na lista?")
print("O gênero mais popular na lista é: ", most_popular_gender(data_list))

# ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
assert type(most_popular_gender(data_list)
            ) is str, "TAREFA 6: Tipo incorreto no retorno. Deveria retornar uma string."
assert most_popular_gender(
    data_list) == "Male", "TAREFA 6: Resultado de retorno incorreto!"
# -----------------------------------------------------

# Se tudo está rodando como esperado, verifique este gráfico!
gender_list = column_to_list(data_list, -2)
types = ["Male", "Female"]
quantity = count_gender(data_list)
y_pos = list(range(len(types)))
plt.bar(y_pos, quantity)
plt.ylabel('Quantidade')
plt.xlabel('Gênero')
plt.xticks(y_pos, types)
plt.title('Quantidade por Gênero')
plt.show(block=True)

input("Aperte Enter para continuar...")

# TAREFA 7
# TODO: Crie um gráfico similar para user_types. Tenha certeza que a legenda está correta.
print("\nTAREFA 7: Verifique o gráfico!")


def count_user_type(data_list, index=5):
    """
    Função que conta user_type da data_list
    Argumentos:
        data_list: Recebe uma lista de todo o arquivo .csv.
        index: Posição da feature que queremos buscar. Default é 5.
    Retorna:
        Uma lista onde o primeiro elemento é a qtd de Subscriber e o segundo elemento é a qtd de Customer.
    """
    subscriber = 0
    customer = 0
    for each in range(len(data_list)):
        if data_list[each][index] == 'Subscriber':
            subscriber += 1
        elif data_list[each][index] == 'Customer':
            customer += 1
    return [subscriber, customer]


user_types_list = column_to_list(data_list, -3)
types = ["Subscriber", "Customer"]
quantity = count_user_type(data_list)
y_pos = list(range(len(types)))
plt.bar(y_pos, quantity)
plt.ylabel('Quantidade')
plt.xlabel('user_types')
plt.xticks(y_pos, types)
plt.title('Quantidade por user_type')
plt.show(block=True)


input("Aperte Enter para continuar...")

# TAREFA 8
# TODO: Responda a seguinte questão
male, female = count_gender(data_list)
print("\nTAREFA 8: Por que a condição a seguir é Falsa?")
print("male + female == len(data_list):", male + female == len(data_list))
answer = "Alguns valores não estão definidos dessa feature."
print("resposta:", answer)

# ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
assert answer != "Escreva sua resposta aqui.", "TAREFA 8: Escreva sua própria resposta!"
# -----------------------------------------------------

input("Aperte Enter para continuar...")

# Vamos trabalhar com trip_duration (duração da viagem) agora. Não conseguimos tirar alguns valores dele.
# TAREFA 9
# TODO: Ache a duração de viagem Mínima, Máxima, Média, e Mediana.
# Você não deve usar funções prontas para isso, como max() e min().
trip_duration_list = column_to_list(data_list, 2)
min_trip = 0.
max_trip = 0.
mean_trip = 0.
median_trip = 0.

# Convertendo os valores strings para int
trip_duration_list_auxiliar = []
for each in range(len(trip_duration_list)):
    trip_duration_list_auxiliar.append(int(trip_duration_list[each]))
trip_duration_list_auxiliar.sort()
trip_duration_list = trip_duration_list_auxiliar

# Encontrando o maior valor através de ifs
for each in range(len(trip_duration_list)):
    if trip_duration_list[each] > max_trip:
        max_trip = trip_duration_list[each]

min_trip = max_trip  # Necessário para percorrer a lista novamente procurando o menor valor

# Encontrando o menor valor através de ifs
for each in range(len(trip_duration_list)):
    if trip_duration_list[each] < min_trip:
        min_trip = trip_duration_list[each]

# Encontrando a média aritmética
total_trip = 0
for each in range(len(trip_duration_list)):
    total_trip += trip_duration_list[each]
mean_trip = round(total_trip / len(trip_duration_list))

# Encontrando a mediana


def mediana(lst):
    """
    Função que retorna a mediana de uma lista
    Argumentos:
        lst: Receba uma lista de valores ints ou floats.
    Retorna:
        A mediana da lst como um número.
    """
    lst_len = len(lst)
    if lst_len < 1:
        return None
    if lst_len % 2 == 1:
        return sorted(lst)[lst_len//2]
    else:
        return sum(sorted(lst)[lst_len//2-1:lst_len//2+1])/2.0


median_trip = round(mediana(trip_duration_list))

print("\nTAREFA 9: Imprimindo o mínimo, máximo, média, e mediana")
print("Min: ", min_trip, "Max: ", max_trip,
      "Média: ", mean_trip, "Mediana: ", median_trip)

# ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
assert round(min_trip) == 60, "TAREFA 9: min_trip com resultado errado!"
assert round(max_trip) == 86338, "TAREFA 9: max_trip com resultado errado!"
assert round(mean_trip) == 940, "TAREFA 9: mean_trip com resultado errado!"
assert round(median_trip) == 670, "TAREFA 9: median_trip com resultado errado!"
# -----------------------------------------------------

input("Aperte Enter para continuar...")

# TAREFA 10
# Gênero é fácil porque nós temos apenas algumas opções. E quanto a start_stations? Quantas opções ele tem?
# TODO: Verifique quantos tipos de start_stations nós temos, usando set()
start_stations_list = []
for each in range(len(data_list)):
    start_stations_list.append(data_list[each][3])
start_stations = set(start_stations_list)

print("\nTAREFA 10: Imprimindo as start stations:")
print(len(start_stations))
print(start_stations)

# ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
assert len(
    start_stations) == 582, "TAREFA 10: Comprimento errado de start stations."
# -----------------------------------------------------

input("Aperte Enter para continuar...")

# TAREFA 11
# Volte e tenha certeza que você documentou suas funções. Explique os parâmetros de entrada, a saída, e o que a função faz. Exemplo:


def new_function(param1: int, param2: str):
    """
    Função de exemplo com anotações.
    Argumentos:
        param1: O primeiro parâmetro.
        param2: O segundo parâmetro.
    Retorna:
        Uma lista de valores x.

    """

input("Aperte Enter para continuar...")

# TAREFA 12 - Desafio! (Opcional)
# TODO: Crie uma função para contar tipos de usuários, sem definir os tipos
# para que nós possamos usar essa função com outra categoria de dados.
print("Você vai encarar o desafio? (yes ou no)")
answer = "yes"


def count_items(column_list):
    item_types = list(set(column_list))
    item_types.sort()
    count_items = []
    for i in range(len(item_types)):
        count_items.append(0)
    for each in range(len(column_list)):
        for i in range(len(item_types)):
            if item_types[i] == column_list[each]:
                count_items[i] += 1
    return item_types, count_items


if answer == "yes":
    # ------------ NÃO MUDE NENHUM CÓDIGO AQUI ------------
    column_list = column_to_list(data_list, -2)
    types, counts = count_items(column_list)
    print("\nTAREFA 12: Imprimindo resultados para count_items()")
    print("Tipos:", types, "Counts:", counts)
    assert len(types) == 3, "TAREFA 12: Há 3 tipos de gênero!"
    assert sum(counts) == 1551505, "TAREFA 12: Resultado de retorno incorreto!"
    # -----------------------------------------------------