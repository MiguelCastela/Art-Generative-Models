# Instalacao e Configuracao

## Versao recomendada de Python

Python 3.10 (recomendado)

Justificacao:
- Boa compatibilidade com PyTorch, torchvision, diffusers, scipy e jupyter
- Evita incompatibilidades frequentes em ambientes academicos com Python 3.12+

---

## Instalacao das dependencias

As instrucoes abaixo assumem que estas na pasta `students/`,
onde se encontra o ficheiro `requirements.txt`.

### Opcao A - Ambiente virtual com venv + pip

1. Criar ambiente virtual

```bash
python3.10 -m venv genai-env
```

2. Ativar o ambiente

- Linux / macOS
```bash
source genai-env/bin/activate
```

- Windows
```bash
genai-env\Scripts\activate
```

3. Instalar dependencias

```bash
pip install -r requirements.txt
```

4. Abrir Jupyter

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```

---

## O que este guiao faz

- Explora `lerp` e `slerp` em espaco de ruido de um DDPM pretreinado
- Calcula `FID` e `KID` sem treinar modelos
- Usa um DDPM pretreinado de MNIST e pesos pretreinados do Inception na primeira execucao
- Todo o codigo esta dentro do notebook, sem ficheiros `.py` auxiliares

Nota:
- A primeira execucao pode descarregar pesos da Hugging Face e do torchvision
- Para a aula usamos poucos exemplos para manter o notebook rapido; os valores de FID/KID ficam mais estaveis com mais amostras
- O exemplo foi mantido leve para uso em aula; se quiseres uma versao mais pesada e realista, podemos trocar depois para CelebA-HQ

---

## Resolucoes rapidas

Erro `ModuleNotFoundError`:
- Verifica se o ambiente virtual esta ativo
- Reinstala com `pip install -r requirements.txt`

Erro ao descarregar pesos:
- Verifica ligacao a internet na primeira execucao
- Se necessario, corre novamente para usar a cache local

GPU nao aparece no PyTorch:
- Verifica o output de `torch.cuda.is_available()`
- Em Apple Silicon, o notebook usa `mps` quando disponivel
