import os
import shutil
from tkinter import Tk, filedialog


def selecionar_pasta(titulo):
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    pasta = filedialog.askdirectory(title=titulo)
    root.destroy()
    return pasta


def mover_arquivos_orfaos(pasta_origem, pasta_destino):
    arquivos = os.listdir(pasta_origem)
    imagens_orfas = []

    for arquivo in arquivos:
        nome_base, ext = os.path.splitext(arquivo)
        ext = ext.lower()

        if ext in [".png", ".jpeg", ".jpg"]:
            txt_correspondente = nome_base + ".txt"
            # Verifica se o TXT correspondente existe (case-insensitive)
            if not any(arq.lower() == txt_correspondente.lower() for arq in arquivos):
                imagens_orfas.append(arquivo)  # Adiciona o nome *original* do arquivo

    for imagem_orfa in imagens_orfas:
        origem = os.path.join(pasta_origem, imagem_orfa)
        destino = os.path.join(pasta_destino, imagem_orfa)
        if os.path.exists(origem):  # Verifica se existe antes de mover
            shutil.move(origem, destino)
            print(f"'{imagem_orfa}' movido para '{pasta_destino}'.")


if __name__ == "__main__":
    pasta_origem = selecionar_pasta("Selecione a pasta de origem")
    pasta_destino = selecionar_pasta("Selecione a pasta de destino")
    if pasta_origem and pasta_destino:
        mover_arquivos_orfaos(pasta_origem, pasta_destino)
    else:
        print("Operação cancelada: pastas de origem e/ou destino não foram selecionadas.")
