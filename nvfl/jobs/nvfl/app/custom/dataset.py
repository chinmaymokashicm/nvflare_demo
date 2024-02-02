import os, re, difflib, json, shutil
from collections import Counter
from rich.progress import track

from pydantic import BaseModel

class DatasetMigration:
    """
    Migrate code from user's folder structure to project
    """
    def __init__(self, folderpath_source_root: str, folderpath_destination_root: str = "data", filename_mappings_json: str ="mappings.json") -> None:
        self.folderpath_source_root = folderpath_source_root
        self.folderpath_destination_root = folderpath_destination_root
        self.dataset_props = DatasetProps()
        self.folderpath_mappings = {os.path.join(level_1, level_2): None for level_1, list_level_2 in self.dataset_props.subfolders.items() for level_2 in list_level_2}
        self.filepath_mappings = None
        self.filename_mappings_json = filename_mappings_json
    
    def _check_folder_structure_validity(self) -> None:
        try:
            dataset_check = DatasetCheck(self.folderpath_source_root)
            dataset_check.check_folder_structure()
            return True
        except Exception as e:
            return False

    def add_foldermap(self, destination_subfolderpath: str, source_subfolderpath: str) -> None:
        self.folderpath_mappings[destination_subfolderpath] = source_subfolderpath

    def _check_filename_consistency(self) -> None:
        """
        Check if filenames are consistent in labels and data subfolders
        """
        if any(value is None for value in self.folderpath_mappings.values()):
            raise Exception(f"""Folder mappings incomplete!
                            {self.folderpath_mappings}""")
        
        dataset_check = DatasetCheck("")
        for level_1 in self.dataset_props.subfolders:
            filenames = []
            subfoldernames = []
            for level_2 in self.dataset_props.subfolders[level_1]:
                filenames.append(os.listdir(os.path.join(self.folderpath_source_root, self.folderpath_mappings[os.path.join(level_1, level_2)])))
                subfoldernames.append(os.path.join(level_1, level_2))
            dataset_check.compare_sublists(filenames, subfoldernames)
        
    def create_filepath_mappings(self) -> list:
        self._check_filename_consistency()
        filenames = [filename for folderpath_source in self.folderpath_mappings.values() for filename in os.listdir(os.path.join(self.folderpath_source_root, folderpath_source))]
        self.source_filenames = sorted(list(set(filenames)))
        self.filepath_mappings = {filename: self.dataset_props.generate_filename(i + 1) for i, filename in enumerate(self.source_filenames)}
    
    def migrate_data(self, make_symlink=True):
        if self.filepath_mappings is None:
            raise Exception("File mappings not yet created!")
        for destination_subfolderpath, source_subfolderpath in self.folderpath_mappings.items():
            destination_folderpath = os.path.join(self.folderpath_destination_root, destination_subfolderpath)
            if not os.path.exists(destination_folderpath):
                os.makedirs(destination_folderpath)
            source_folderpath = os.path.join(self.folderpath_source_root, source_subfolderpath)
            for source_filename in track(os.listdir(source_folderpath), description=f"From {source_folderpath} to {destination_folderpath}"):
                destination_filename = self.filepath_mappings[source_filename]
                source_filepath = os.path.join(source_folderpath, source_filename)
                destination_filepath = os.path.join(destination_folderpath, destination_filename)
                if os.path.exists(destination_filepath):
                    print(f"{destination_filepath} already exists. Skipping...")
                    continue
                if make_symlink:
                    os.symlink(source_filepath, destination_filepath)
                else:
                    shutil.copy2(source_filepath, destination_filepath)
        with open(os.path.join(self.folderpath_destination_root, self.filename_mappings_json), "w") as f:
            json.dump(self.filepath_mappings, f, indent=4, sort_keys=True)

class DatasetProps(BaseModel):
    prefix: str = "uth"
    separator: str = "_"
    file_extension: str = ".nii.gz"
    subfolders: dict = {
        "train": ["data", "labels"],
        "test": ["data", "labels"]
    }
    n_leading_zeros: int = 3
    
    def generate_filename(self, number) -> str:
        prefix = self.prefix
        separator = self.separator
        extension = self.file_extension
        n_leading_zeros = self.n_leading_zeros
        return f"{prefix}{separator}{str(number).zfill(n_leading_zeros)}{extension}"
    
class DatasetCheck:
    def __init__(self, folderpath: str) -> None:
        self.dataset_props = DatasetProps()
        self.folderpath: str = folderpath
        self.filename_template: str = self._generate_filename_template()
        # self.check_folder_structure()
        # self.filenames = self.check_files_nomenclature()
    
    def check_folder_structure(self) -> bool:
        level_1_subfolders: list[str] = next(os.walk(self.folderpath))[1]
        if Counter(level_1_subfolders) != Counter(self.dataset_props.subfolders.keys()):
            for foldername in self.dataset_props.subfolders.keys():
                if foldername not in level_1_subfolders:
                    raise Exception(f"{foldername} not found in {level_1_subfolders}")
        for level_1, list_level_2 in self.dataset_props.subfolders.items():
            if level_1 in level_1_subfolders:
                level_2_subfolders = next(os.walk(os.path.join(self.folderpath, level_1)))[1]
                if Counter(level_2_subfolders) != Counter(list_level_2):
                    for foldername in list_level_2:
                        if foldername not in level_2_subfolders:
                            raise Exception(f"{foldername} not found in {level_1}")
        return True
    
    def _generate_filename_template(self) -> str:
        prefix = self.dataset_props.prefix
        separator = self.dataset_props.separator
        extension = self.dataset_props.file_extension
        return f"^{re.escape(prefix)}{re.escape(separator)}\d{{{self.dataset_props.n_leading_zeros}}}{re.escape(extension)}$"
    
    def _validate_string(self, string) -> bool:
        pattern: re.Pattern = re.compile(self.filename_template)
        match: re.Match | None = pattern.match(string)
        return bool(match)
    
    def compare_sublists(self, list_of_lists: list, listnames: list = None) -> None:
        difflib_obj = difflib.Differ()
        for i, level_2_i in enumerate(list_of_lists):
            for j, level_2_j in enumerate(list_of_lists):
                if i < j:
                    diff = difflib_obj.compare(level_2_i, level_2_j)
                    if any(line.startswith("-") for line in diff):
                        display_value_i, display_value_j = i, j
                        if listnames is not None:
                            display_value_i, display_value_j = listnames[i], listnames[j]
                        error_statement_0 = f"Looks like not all files are common between {display_value_i} and {display_value_j}!"
                        error_statement_1 = f"Differences between {display_value_i} and {display_value_j}:"
                        print("\n".join([error_statement_0, error_statement_1]))
                        for line in diff:
                            if line.startswith("-") or line.startswith("+"):
                                print(line)
                        raise Exception("Filenames not matching!")
    
    def check_files_nomenclature(self) -> dict:
        filenames: dict = {}
        for level_1_foldername, list_level_2_foldernames in self.dataset_props.subfolders.items():
            if level_1_foldername not in filenames:
                filenames[level_1_foldername] = {}
            for level_2_foldername in list_level_2_foldernames:
                if level_2_foldername not in filenames[level_1_foldername]:
                    filenames[level_1_foldername][level_2_foldername] = []
                for filename in os.listdir(os.path.join(self.folderpath, level_1_foldername, level_2_foldername)):
                    if not self._validate_string(filename):
                        raise Exception(f"{filename} in {os.path.join(level_1_foldername, level_2_foldername)} is of the wrong format. It should be in the format {self.filename_template}")
                    if filename in filenames[level_1_foldername][level_2_foldername]:
                        raise Exception(f"Is {filename} repeated in {os.path.join(self.folderpath, level_1_foldername, level_2_foldername)}?")
                    filenames[level_1_foldername][level_2_foldername].append(filename)
        
        # Verify that each set has the appropriate numbers
        for level_1 in filenames:
            list_of_lists: list = []
            listnames: list = []
            for level_2 in filenames[level_1]:
                list_of_lists.append(filenames[level_1][level_2])
                listnames.append(level_2)
            self.compare_sublists(list_of_lists, listnames=listnames)
            
        return filenames
    
    def _check_file_properties(self):
        if self.filenames is None:
            raise Exception("Filenames not available!")
    
if __name__=="__main__":
    # dataset_check = DatasetCheck("test_data")
    # print(dataset_check.check_folder_structure())
    # print(dataset_check.validate_string("uth_002.nii.gz"))
    # print(dataset_check.validate_string("uth_012"))
    # print(dataset_check.validate_string("uth_102"))
    # print(dataset_check.validate_string("uth_000.nii.gz"))
    # print(dataset_check.validate_string("uth_100"))
    # print(dataset_check.validate_string("uth_10.nii.gz"))
    
    # lists = [
    #     [1, 2, 3, 4],
    #     [1, 2, 4, 5],
    #     [2, 3, 4, 5],
    #     [2, 3, 4, 5]
    # ]
    # dataset_check.compare_sublists(lists)
    
    dataset_migration = DatasetMigration("/data/cmokashi/msd/brain_tumor")
    dataset_migration.folderpath_mappings = {
        "train/data": "imagesTr", "train/labels": "labelsTr", "test/data": "imagesTr", "test/labels": "labelsTr"
    }
    dataset_migration.create_filepath_mappings()
    # print(dataset_migration.filepath_mappings)
    dataset_migration.migrate_data(make_symlink=True)
    
    