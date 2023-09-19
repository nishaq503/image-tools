"""Helper functions for the Image Registration plugin."""

import itertools
import pathlib

import numpy as np
from filepattern import FilePattern
from filepattern import get_regex
from filepattern import parse_directory


def parse_collection(
    directory_path: pathlib.Path,
    file_pattern: str,
    registration_variable: str,
    similarity_variable: str,
    template_image: np.ndarray,  # noqa: ARG001
) -> dict[tuple[np.ndarray, np.ndarray], list[np.ndarray]]:
    """This function parses the input directory and returns a dictionary.

    Each key in the dictionary is a tuple consisting of a template and a moving image.
    The value corresponding to each key is a list of images that have similar
    transformation as the moving image.

    Note: The code produces the expected output when
    len(registration_variable)==len(similarity_variable)==1. The code will NOT
    spit out an error when the more than one variable is passed as registration
    or similarity variable, but additional testing needs to be done to validate
    the script for this use case.

    Args:
        directory_path: path to the input collection
        file_pattern: file name pattern of the input images
        registration_variable: variable to help determine the set of moving and
            template images
        similarity_variable: variable to help determine the set of images having
            a similar transformation corresponding to each set of moving and
            template images
        template_image: name of a template image

    outputs : result_dic
              example of result_dic is shown below

    result_dic = {(template_img1: moving_img1): [set1_img1,set1_img2, set1_img3....],
                  (template_img2: moving_img2): [set2_img1,set2_img2, set2_img3....],
                                              .
                                              .                                     }
    """
    # Predefined variables order

    # get all variables in the file pattern
    _, variables = get_regex(file_pattern)

    # get variables except the registration and similarity variable
    moving_variables = [
        var
        for var in variables
        if var not in registration_variable + similarity_variable
    ]

    # uvals is dictionary with all the possible variables as key
    # corresponding to each key is a list of all values which that variable can
    # take for the input collection
    _, uvals = parse_directory(directory_path, file_pattern)

    parser_object = FilePattern(directory_path, file_pattern)

    image_set = []

    # extract the index values from uvals for each variable in moving_variables
    moving_variables_set = [uvals[var] for var in moving_variables]

    # iterate over the similar transformation variables
    # Code produced expected output when
    # refer to function description
    for char in similarity_variable:
        # append the variable to the moving variable set
        moving_variables.append(char)

        # iterate over all possible index values of the similar transf. variable
        for ind in uvals[char]:
            registration_set = []

            # append the fixed value of the index to the moving variables set
            moving_variables_set.append([ind])

            # get all the possible combinations of the index values in the moving
            # variables set
            registration_indices_combinations = list(
                itertools.product(*moving_variables_set),
            )
            all_dicts = []

            # iterate over all combinations and create a dictionary for each combination
            # the dictionary is of the form {'C'=1, 'X'=2...etc} which can be used as
            # an input to the get_matching() function
            for index_comb in registration_indices_combinations:
                inter_dict = {}
                for i in range(len(moving_variables)):
                    inter_dict.update({moving_variables[i].upper(): index_comb[i]})
                # store all dictionaries
                all_dicts.append(inter_dict)

            # iterate over all dictionaries
            for reg_dict in all_dicts:
                intermediate_set = []
                # use get_matching function to get all filenames with defined
                # variable values in the dictionary
                files = parser_object.get_matching(**reg_dict)

                # files is a list of dictionaries
                for file_dict in files:
                    intermediate_set.append(file_dict["file"])
                registration_set.append(intermediate_set)

            # delete the fixed index value of the similar transf.
            # variable to prepare for the next iteration
            moving_variables_set.pop(-1)
            image_set.append(registration_set)

    # parse image set to form the result dictionary
    result_dic = {}
    old_set = np.array(image_set)
    for j in range(old_set.shape[1]):
        inter = old_set[:, j, :]
        for k in range(inter.shape[1]):
            ky = (inter[0, 0], inter[0, k])
            items = inter[1:, k]
            result_dic.update({ky: items})

    return result_dic
