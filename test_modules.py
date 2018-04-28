from Helpers import path_utils
from Helpers import deployment_utils as deploy
import os
import pytest

class Test_Answer_Sentence_Selection:

    @staticmethod
    def test_preprocess():
        return

    @staticmethod
    def test_passage_retrieval():
        return

    @staticmethod
    def test_sentence_ranking():
        return

    @staticmethod
    def test_preprocess():
        return


class Test_Configuration:
    test_config_filename = 'test.cfg'

    @staticmethod
    def get_test_config_file():
        filename = Test_Configuration.test_config_filename
        CONFIG_PATH = path_utils.get_config_path()
        filepath = os.path.join(CONFIG_PATH, filename)
        return filepath

    @staticmethod
    def del_test_config_file():
        filepath = Test_Configuration.get_test_config_file()
        os.remove(filepath)
        return

    @staticmethod
    def test_get_config_valid():
        filepath = Test_Configuration.get_test_config_file()
        with open(filepath,'w') as f:
            f.write('name=meshde\n')
            f.write('age=22\n')
            f.write('stud=yes\n')

        config = deploy.get_config(Test_Configuration.test_config_filename)

        for key in ['name', 'age', 'stud']:
            assert(key in config)
        assert isinstance(config['age'], int)

        Test_Configuration.del_test_config_file()
        return

    @staticmethod
    def test_get_config_file_not_found(capsys):
        filepath = Test_Configuration.get_test_config_file()

        with pytest.raises(FileNotFoundError):
            config = deploy.get_config(Test_Configuration.test_config_filename)

        # captured = capsys.readouterr()
        # required_output = '{0} has not been created yet!'.format(
        #     Test_Configuration.test_config_filename,
        # )

        # assert captured.out == required_output
        return

    @staticmethod
    def test_get_config_invalid_format(capsys):
        filepath = Test_Configuration.get_test_config_file()
        with open(filepath,'w') as f:
            f.write('name:meshde\n')
            f.write('age:22\n')
            f.write('stud:yes\n')

        with pytest.raises(ValueError):
            config = deploy.get_config(Test_Configuration.test_config_filename)

        # captured = capsys.readouterr()
        # required_output = 'The contents of file {} are not in proper format\n'
        # 'Format: key=value'.format(
        #     Test_Configuration.test_config_filename,
        # )

        # assert captured.out == required_output
        return

