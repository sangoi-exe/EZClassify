import shutil
from pathlib import Path
from tkinter import Tk, filedialog


def selecionar_pasta(titulo):
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    pasta = filedialog.askdirectory(title=titulo)
    root.destroy()
    return Path(pasta) if pasta else None


def mover_arquivos_orfaos(pasta_origem, pasta_destino):
    # Obtém todos os itens (arquivos e pastas) da pasta de origem
    itens = list(pasta_origem.iterdir())
    # Cria um dicionário para busca case-insensitive: nome do arquivo em minúsculas -> objeto Path
    arquivos_lower = {item.name.lower(): item for item in itens if item.is_file()}

    imagens_orfas = []
    txt_orfas = []

    # Verifica cada arquivo para identificar órfãos
    for item in itens:
        if not item.is_file():
            continue

        nome_base = item.stem  # Nome sem extensão
        ext = item.suffix.lower()  # Extensão em minúsculas

        # Se for imagem, verifica se existe o .txt correspondente
        if ext in [".png", ".jpeg", ".jpg"]:
            txt_nome = nome_base + ".txt"
            if txt_nome.lower() not in arquivos_lower:
                imagens_orfas.append(item)
        # Se for arquivo de texto, verifica se existe alguma imagem com o mesmo nome base
        elif ext == ".txt":
            encontrou_imagem = False
            for image_ext in [".png", ".jpeg", ".jpg"]:
                candidato = nome_base + image_ext
                if candidato.lower() in arquivos_lower:
                    encontrou_imagem = True
                    break
            if not encontrou_imagem:
                txt_orfas.append(item)

    # Move as imagens órfãs
    for imagem_orfa in imagens_orfas:
        destino = pasta_destino / imagem_orfa.name
        if imagem_orfa.exists():
            shutil.move(str(imagem_orfa), str(destino))
            print(f"Imagem órfã '{imagem_orfa.name}' movida para '{pasta_destino}'.")

    # Move os arquivos de texto órfãos
    for txt_orf in txt_orfas:
        destino = pasta_destino / txt_orf.name
        if txt_orf.exists():
            shutil.move(str(txt_orf), str(destino))
            print(f"Arquivo de texto órfão '{txt_orf.name}' movido para '{pasta_destino}'.")


if __name__ == "__main__":
    pasta_origem = selecionar_pasta("Selecione a pasta de origem")
    pasta_destino = selecionar_pasta("Selecione a pasta de destino")
    if pasta_origem and pasta_destino:
        mover_arquivos_orfaos(pasta_origem, pasta_destino)
    else:
        print("Operação cancelada: pastas de origem e/ou destino não foram selecionadas.")
