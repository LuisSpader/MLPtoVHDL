# Truncate Neural Network
## Por que *truncar* os números?
Em linguagens de alto nível, os números geralmente são do tipo float, podendo ser de [precisão simples ou precisão dupla](https://learn.microsoft.com/pt-br/cpp/build/ieee-floating-point-representation?view=msvc-170) (32 bits e 64 bits respectivamente). A ideia de utilizar números truncados, i.e. menos bits, tem como objetivo trazer para o hardware uma forma menos custosa de implementar os cálculos de Multiplica e Acumula (MAC), operação padrão para o cálculo de redes neurais.

![Perceptron](https://embarcados.com.br/wp-content/uploads/2016/09/Perceptron-01.png)

## Setup
O arquivo requirements.txt possui todos os imports necessários para instalação, utilizando o comando 
```console
foo@bar:~$ pip install -r requirements.txt
```

você instalará todas as dependências necessárias para rodar o arquivo. Caso queira utilizar um ambiente virtual, o [Pipenv](https://pipenv.pypa.io/en/latest/) ou [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) são algumas das possíveis opções. A vantagem de utilizar um ambiente virtual é de não ter que ficar trocando as versões dos imports do python.

## Estrutura do trabalho
Os dois principais arquivos são: truncate_parallel e truncate_sequential.

<ol>
    <li>truncate_parallel 
        <p> É o algoritmo que faz os cálculos de maneira mais rápida, e deve ser utilizado para geração de resultados de precisão do modelo truncado em comparação com o modelo treinado pelo <i>tensorflow</i>.
        </p>
    </li>
    <li>truncate_sequential
        <p>Este é utilizado como um <i>debug</i>, pois é mais lento e gera um arquivo com os valores intermediários dos MACs, servindo para verificar </p>
    </li>
</ol>

## Como utilizar
```
###     CONFIGURATION    ####
# Passe as configurações do modelo que estão salvas na pasta /models.

# O layers está ligado com o número de layers, é a forma em que temos salvado
# os modelos,
layers = "4_4_4_4_4_4_4_4_4"  
n_layers = "10"
neurons = "4"   # Número de neurônios do modelo treinado


# Mude os diretórios de acordo com o seu modelo, MNIST/4X4/LAYERS...
# significa que o modelo é o MNIST, as imagens são 4X4 e o que os modelos
# variam um do outro é a quantidade de camadas.
model = tf.keras.models.load_model(f'{path}/models/MNIST/4X4/LAYERS/{neurons}_NEURONS/NN_{n_layers}Layers_16_{layers}_10.h5')
print("\n\n\n\n")
B = 12 # Number of bits
write_path = f'{path}/model_results/ACCURACY/4X4/LAYERS/{neurons}_NEURONS/2_bitsmul/NN_{n_layers}Layers_{B}bits_16_{layers}_10.txt'
```

Você deve modificar as configurações acima. Segue uma descrição mais detalhada:
<ul>
    <li>
        <i>layers</i> é simplesmente uma convenção que eu e o <a href="https://github.com/LuisSpader">Luis Spader</a> tivemos, sendo que você pode identificar como quiser de modo que fique explícito as camadas do meio. Como o número de layers já está explícito na linha seguinte, essa variável não é estritamente necessária.
    </li>
    <li>
        <i>n_layers</i> é autoexplicativo, uma vez que diz respeito ao número de camadas <strong>sem</strong> contar o input, ou seja, são todas as camadas intermediárias + a última camada.
    </li>
    <li>
        <i>neurons</i> como está comentado no código, é o número de neurônios por camada.
    </li>
    <li>
        <i>model</i> é o diretório onde você deve carregar o modelo. Há uma certa automatização nesse processo já que você só deve mudar as variáveis, mas ainda assim você pode salvar em qualquer outro lugar os modelos e carregá-los de forma diferente
    </li>
    <li>
        <i>B</i> é o número de bits que o código irá utilizar como precisão para calcular o valor do modelo
    </li>
    <li>
        <i>write_path</i> é onde você quer que os resultados sejam escritos.
    </li>
</ul>
