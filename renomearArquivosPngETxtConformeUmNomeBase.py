import logging
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Set, Tuple

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FileRenamer:
    def __init__(self, base_name: str):
        self.base_name = base_name
        self.counter = 1

    def select_directory(self) -> Path:
        # Abre janela para selecionar o diretório
        root = Tk()
        root.withdraw()  # Esconde a janela principal do Tk
        root.attributes("-topmost", True)  # Mantém a janela de seleção no topo
        try:
            directory = filedialog.askdirectory(title="Selecione o diretório para renomear")
            if directory:
                return Path(directory)
            else:
                logging.warning("Nenhum diretório selecionado.")
                raise ValueError("Nenhum diretório selecionado.")
        finally:
            root.destroy()  # Garante que a janela do Tk seja destruída

    def rename_files(self, directory: Path) -> None:
        logging.info(f"Iniciando renomeação no diretório: {directory}")

        files = [item for item in directory.iterdir() if item.is_file()]
        subdirectories = [item for item in directory.iterdir() if item.is_dir()]

        # Categorizar arquivos por extensão
        files_by_extension: dict[str, Set[Path]] = {}
        for file in files:
            files_by_extension.setdefault(file.suffix.lower(), set()).add(file)

        # Agregar arquivos de imagem suportados (.png, .jpg, .jpeg)
        image_files = set()
        for ext in [".png", ".jpg", ".jpeg"]:
            image_files |= files_by_extension.get(ext, set())
        txt_files = files_by_extension.get(".txt", set())
        json_files = files_by_extension.get(".json", set())
        npz_files = files_by_extension.get(".npz", set())

        # Excluir arquivos de máscara da lista de imagens principais
        image_files = {file for file in image_files if not file.stem.endswith("-masklabel")}

        # Encontrar pares de arquivos de imagem e TXT com o mesmo nome base
        pairs = self.find_pairs(image_files, txt_files)

        # Ordenar os pares pelo nome do arquivo de imagem original
        sorted_pairs = sorted(pairs, key=lambda x: x[0].name)

        # Renomear os arquivos pareados
        for image_file, txt_file, json_file, npz_file in sorted_pairs:
            new_base_name = f"{self.base_name}_{str(self.counter).zfill(3)}"
            try:
                # Renomear imagem mantendo a extensão original
                new_image_name = f"{new_base_name}{image_file.suffix}"
                self.rename_file(image_file, directory / new_image_name)
                # Renomear arquivo TXT
                self.rename_file(txt_file, directory / f"{new_base_name}.txt")

                # Renomear arquivo JSON, se existir
                if json_file and json_file in json_files:
                    self.rename_file(json_file, directory / f"{new_base_name}.json")

                # Renomear arquivo NPZ, se existente
                if npz_file and npz_file in npz_files:
                    self.rename_file(npz_file, directory / f"{new_base_name}.npz")

                # Verificar e renomear o arquivo de máscara correspondente
                mask_suffix = f"-masklabel{image_file.suffix}"
                original_mask_name = image_file.stem + mask_suffix
                original_mask_file = directory / original_mask_name
                if original_mask_file.exists():
                    new_mask_name = f"{new_base_name}{mask_suffix}"
                    self.rename_file(original_mask_file, directory / new_mask_name)
                else:
                    logging.info(f"Arquivo de máscara não encontrado para: {image_file.name}")

                self.counter += 1
            except Exception as e:
                logging.error(f"Erro ao renomear arquivos: {e}")

        # Processar subdiretórios recursivamente
        for subdir in subdirectories:
            self.rename_files(subdir)

    def find_pairs(self, image_files: Set[Path], txt_files: Set[Path]) -> Set[Tuple[Path, Path, Path, Path]]:
        # Encontra pares de arquivos de imagem e TXT com o mesmo nome base
        pairs = set()
        txt_names = {file.stem: file for file in txt_files}

        for image_file in image_files:
            base_name = image_file.stem
            txt_file = txt_names.get(base_name)
            if txt_file:
                json_file = image_file.with_suffix(".json")
                npz_file = image_file.with_suffix(".npz")
                pairs.add((image_file, txt_file, json_file, npz_file))
        return pairs

    def rename_file(self, original: Path, new: Path) -> None:
        # Renomeia um arquivo de original para novo
        if new.exists():
            logging.warning(f"O arquivo {new} já existe e será sobrescrito.")
        original.rename(new)
        logging.info(f"Renomeado: {original.name} para {new.name}")


def main():
    base_name = "Docs-v1"  # Ajuste conforme necessário
    renamer = FileRenamer(base_name)

    try:
        directory = renamer.select_directory()
        renamer.rename_files(directory)
        logging.info("Renomeação concluída com sucesso.")
    except Exception as e:
        logging.error(f"O processo de renomeação falhou: {e}")


if __name__ == "__main__":
    main()
