# MPEI_CanvaCheck


Mª Inês Gonçalves - 124754

Íris Lucas - 125348

## Structure of repository Breakdown

### 1. `modules/`
Contem o código de cada algoritmo (MinHash, Bloom Filter, Naive Bayes) separado, para facilitar os testes.

- `minHash.m` : Funções para dividir as imagens em patches mais pequenos e calcular a similaridade de Jaccard.

- `bloomFilter.m` : Lida com a fase de treino (inserir os patches das imagens reais) e a fase de teste (ver se é real ou não).

- `naiveBayes_train.m` : Extraí 5 features principais: histograma de cor, textura (GLCM), edge density, MinHash score, e o Bloom Filter ratio.

### 2. `tests/`

Contem os testes para cada módulo.

### 3. `data/`

Contem imagens para serem usadas:

- `authentic/` : contem as imagens reais que vão servir de referência.

- `test` : contem uma mistura de imagens, reais e falsas, usadas para demonstrar a aplicação

### 4. `report/`

Contem o report do projeto

### 5. `main.m`

Script principal, que usa as imagens (a real, e as misturadas) e faz a "decisão":

1. Usa uma imagem de referência e de test como input.

2. Corre os módulos de MinHash e do Bloom Filter para conseguir as pontuações

3. Mete as pontuações adquiridas no classificador de Naives Bayes.

4. Output: Mostra as imagens lado a lado com o  resultado: "Esta pintura é x% de ser Falsa" 

## How to run the program



## Test Suite