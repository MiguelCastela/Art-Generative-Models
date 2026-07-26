# Instalacao e Configuracao

## Versao recomendada de Python

Python 3.10 (recomendado)

Justificacao:
- Totalmente compativel com pytorch, torchvision, numpy, matplotlib e jupyter
- Estavel e amplamente suportado em ambientes academicos
- Evita incompatibilidades ainda frequentes em Python 3.11+ e 3.12

---

## Instalacao das dependencias

As instrucoes abaixo assumem que estas na pasta do projeto,
onde se encontra o ficheiro requirements.txt.

### Opcao A - Ambiente virtual com venv + pip (recomendado)

1) Criar ambiente virtual

```bash
python3.10 -m venv genai-env
```

2) Ativar o ambiente

- Linux / macOS
```bash
source genai-env/bin/activate
```

- Windows
```bash
genai-env\Scripts\activate
```

3) Instalar dependencias

```bash
pip install -r requirements.txt
```

4) Abrir Jupyter

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```

---

## Resolucoes rapidas

Erro ModuleNotFoundError:
- Verifica se o ambiente virtual esta ativo
- Reinstala com pip install -r requirements.txt no ambiente correto

GPU nao aparece no PyTorch:
- Verifica o output de `torch.cuda.is_available()`
- Se necessario, instala a versao de PyTorch adequada ao sistema

Graficos nao aparecem:
- Adiciona `%matplotlib inline` no topo do notebook
