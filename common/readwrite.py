'''
Created on Dec 6, 2020

@author: smin
'''
import json
import pickle
import pandas as pd


class ReadFile(object):
    """
    Contains methods for reading various file types
    """

    @staticmethod
    def _print_read(filepath):
        print('(read) %s' % filepath)

    @staticmethod
    def read_csv(filepath):
        """
        Read CSV file
        :param filepath:
        :return:
        """
        ReadFile._print_read(filepath)

        pass

    @staticmethod
    def read_csvs(filepaths, verbose=False):
        """
        Read list of CSV files into single dataframe
        :param filepaths:
        :return:
        """
        if verbose:
            ReadFile._print_read(filepaths)

        dfs = [pd.read_csv(filepath) for filepath in filepaths]
        df = pd.concat(dfs)
        return df

    @staticmethod
    def read_json(filepath, verbose=False):
        """
        Read JSON file
        :param print_read:
        :param filepath:
        :return: json content
        """
        if verbose:
            ReadFile._print_read(filepath)

        with open(filepath, 'r') as f:
            try:
                json_file = json.load(f)
            except ValueError:
                json_file = {}
                print('ERROR - not a valid json')

        return json_file

    @staticmethod
    def read_lines(filepath, verbose=False):
        """
        Read file line by line
        :param filepath:
        :return: list of rows
        """
        if verbose:
            ReadFile._print_read(filepath)

        with open(filepath, 'r') as f:
            lines = f.read().splitlines()

        return lines

    @staticmethod
    def read_pickle(filepath):
        """
        read a pickle file
        :param filepath:
        :return:
        """
        temp = open(filepath, 'r')
        pickle_file = pickle.load(temp)
        temp.close()

        return pickle_file


class WriteFile(object):
    """
    Contains methods for writing to various file types
    """

    @staticmethod
    def _print_write(filepath):
        print('(write) %s' % filepath)

    @staticmethod
    def write_file(filepath, content, verbose=False):
        """
        Write content to file
        :param filepath:
        :param content:
        :return:
        """
        if verbose:
            WriteFile._print_write(filepath)

        with open(filepath, 'w') as f:
            f.write(content)

    @staticmethod
    def write_json(filepath, content, verbose=False):
        """
        Write content to json
        :param filepath:
        :param content:
        :return:
        """
        from json import dumps

        if verbose:
            WriteFile._print_write(filepath)

        with open(filepath, 'w') as f:
            f.write(dumps(content, indent=4))
            f.write('\n')

    @staticmethod
    def df_to_csv(
            filepath, df, date_format='%Y%m%d',
            index_bool=True, append_mode=False, header_bool=True, verbose=False):
        """
        Write dataframe to csv file
        :param filepath:
        :param df:
        :param date_format:
        :param index_bool:
        :param header_bool:
        :param append_mode:
        :return:
        """
        if verbose:
            WriteFile._print_write(filepath)

        if index_bool:
            df = df.reset_index()
            index_bool = False

        if append_mode:
            mode_ = 'a'
        else:
            mode_ = 'w'

        with open(filepath, mode_) as f:
            df.to_csv(
                f, header=header_bool, index=index_bool, sep=',',
                date_format=date_format
            )

    @staticmethod
    def write_pickle(item_to_store, filepath):
        """
        store item to a pickle
        :param item_to_store:
        :param filepath:
        :return:
        """
        temp = open(filepath, 'wb')
        pickle.dump(item_to_store, temp)
        temp.close()
