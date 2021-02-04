from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
from sklearn.utils import check_array
import numpy as np
import typing
import time
import pandas as pd
import copy


from numpy.random import RandomState
from math import ceil
from numba import njit
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import (
    check_array, check_is_fitted, check_random_state)

from d3m import container
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase


__all__ = ('RandomSubsequenceSegmentation',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]
    # indices_: Optional[ndarray]
    clf_: Optional[BaseEstimator]

class Hyperparams(hyperparams.Hyperparams):
    n_windows = hyperparams.Union[Union[int, float]](
        configuration=OrderedDict(
            length=hyperparams.Hyperparameter[int](
                default=1,
            ),
            percentage=hyperparams.Hyperparameter[float](
                default=1.0,
            ),
        ),
        default='percentage',
        description='The number of windows from which features are extracted.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    min_window_size = hyperparams.Union[Union[int, float]](
        configuration=OrderedDict(
            length=hyperparams.Hyperparameter[int](
                default=1,
            ),
            percentage=hyperparams.Hyperparameter[float](
                default=1.0,
            ),
        ),
        default='length',
        description='The minimum length of the windows. If float, it represents a percentage of the size of each time series.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    random_state = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            rint=hyperparams.Hyperparameter[int](
                default=1,
            ),
            rnone=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='rnone',
        description='The seed of the pseudo random number generator to use when shuffling\
                    the data. If int, random_state is the seed used by the random number\
                    generator. If RandomState instance, random_state is the random number\
                    generator. If None, the random number generator is the RandomState\
                    instance used by `np.random`.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )


    columns_using_method= hyperparams.Enumeration(
        values=['name', 'index'],
        default='index',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Choose to use columns by names or indecies. If 'name', \"use_columns\" or \"exclude_columns\" is used. If 'index', \"use_columns_name\" or \"exclude_columns_name\" is used."
    )
    use_columns_name = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column names to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns_name = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column names to not operate on. Applicable only if \"use_columns_name\" is not provided.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )  
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    error_on_no_input = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.",
    )
    
    return_semantic_type = hyperparams.Enumeration[str](
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    
class RandomSubsequenceSegmentation(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Random Subsequence Time Seires Segmentation.

    Parameters
    ----------
    n_windows : int or float (default = 1.)
        The number of windows from which features are extracted.
    min_window_size : int or float (default = 1)
        The minimum length of the windows. If float, it represents a percentage
        of the size of each time series.
    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    use_columns: Set
        A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.
    
    exclude_columns: Set
        A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.
    
    return_result: Enumeration
        Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.
    
    use_semantic_types: Bool
        Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe.
    
    add_index_columns: Bool
        Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".
    
    error_on_no_input: Bool(
        Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.
    
    return_semantic_type: Enumeration[str](
        Decides what semantic type to attach to generated attributes'
    """

    __author__: "DATA Lab at Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({ 
         "name": "Subsequence Clustering Primitive",
         "python_path": "d3m.primitives.tods.feature_analysis.random_subsequence_segmentation",
         "source": {'name': 'DATA Lab at Texas A&M University', 'contact': 'mailto:khlai037@tamu.edu', 
         'uris': ['https://gitlab.com/lhenry15/tods.git', ]},
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.RANDOM_SUBSEQUENCE_SEGMENTATION,],
         "primitive_family": metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
         "id": "89c4adff-6656-4d05-8d5f-95c1f70bbd6b",
         "hyperparams_to_tune": ['n_windows', 'min_window_size', 'random_state'],
         "version": "0.0.1",
    })


    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = WindowFeatureExtractor(
              n_windows=self.hyperparams['n_windows'],
              min_window_size=self.hyperparams['min_window_size'],
              random_state=self.hyperparams['random_state'],
        )      
        
        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._fitted = False
        self.indices_ = None
        
        
    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Set training data for SKTruncatedSVD.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        # self.logger.warning('set was called!')
        self._inputs = inputs
        self._fitted = False
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """
        if self._fitted: # pragma: no cover
            return CallResult(None)

        # # Get cols to fit.
        # self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        # self._input_column_names = self._training_inputs.columns

        # If there is no cols to fit, return None
        if self._inputs is None:
            return CallResult(None)

        

        # if len(self._training_indices) > 0:
        if self._inputs is not None:
            self._clf.fit(self._inputs)

            self._fitted = True
            print(self._fitted)

        else: # pragma: no cover
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame.

        Returns:
            Container DataFrame after Truncated SVD.
        """

        if not self._fitted: # pragma: no cover
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs
        # if self.hyperparams['use_semantic_types']:
        #     sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(sk_inputs) > 0:
            sk_output = self._clf.transform(sk_inputs)
            # if sparse.issparse(sk_output):
            #     sk_output = sk_output.toarray()
            print(sk_output)
            sk_output = pd.DataFrame(sk_output)
            
            # outputs = self._wrap_predictions(inputs, sk_output)
            # if len(outputs.columns) == len(self._input_column_names):
            #     outputs.columns = self._input_column_names
            # output_columns = [outputs]
        # else: # pragma: no cover
        #     if self.hyperparams['error_on_no_input']:
        #         raise RuntimeError("No input columns were selected")
        #     self.logger.warn("No input columns were selected")
        # outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
        #                                        add_index_columns=self.hyperparams['add_index_columns'],
        #                                        inputs=inputs, column_indices=self._training_indices,
        #                                        columns_list=output_columns)

        return CallResult(sk_output)

    def get_params(self) -> Params:
        """
        Return parameters.
        Args:
            None
        Returns:
            class Params
        """

        if not self._fitted:
            return Params(
                # lambda_=None,
                clf_ = copy.copy(self._clf),
                # Keep previous
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            # lambda_=getattr(self._clf, 'lambda_', None),
            clf_=copy.copy(self._clf),
            # Keep previous
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )


    def set_params(self, *, params: Params) -> None:
        """
        Set parameters for Powertransformer.
        Args:
            params: class Params
        Returns:
            None
        """

        # self._clf.lambda_ = params['lambda_']
        self._clf = params['clf_']
        # Keep previous
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']

        # if params['lambda_'] is not None:
        #     self._fitted = True
        if params['clf_'] is not None:
            self._fitted = True
        
    
    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        """
        Select columns to fit.
        Args:
            inputs: Container DataFrame
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            list
        """
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        use_columns = []
        exclude_columns = []

        # if hyperparams['columns_using_method'] == 'name':
        #     inputs_cols = inputs.columns.values.tolist()
        #     for i in range(len(inputs_cols)):
        #         if inputs_cols[i] in hyperparams['use_columns_name']:
        #             use_columns.append(i)
        #         elif inputs_cols[i] in hyperparams['exclude_columns_name']:
        #             exclude_columns.append(i)      
        # else: 
        use_columns=hyperparams['use_columns']
        exclude_columns=hyperparams['exclude_columns']           
        
        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata, use_columns=use_columns, exclude_columns=exclude_columns, can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
        """
        Output whether a column can be processed.
        Args:
            inputs_metadata: d3m.metadata.base.DataMetadata
            column_index: int

        Returns:
            boolnp
        """
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, np.integer, np.float64)
        accepted_semantic_types = set()
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        
        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False
    
    
    @classmethod
    def _update_predictions_metadata(cls, inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
                                     target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:
        """
        Updata metadata for selected columns.
        Args:
            inputs_metadata: metadata_base.DataMetadata
            outputs: Container Dataframe
            target_columns_metadata: list

        Returns:
            d3m.metadata.base.DataMetadata
        """
        outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            column_metadata.pop("structural_type", None)
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:
        """
        Wrap predictions into dataframe
        Args:
            inputs: Container Dataframe
            predictions: array-like data (n_samples, n_features)

        Returns:
            Dataframe
        """
        outputs = d3m_dataframe(predictions, generate_metadata=True)
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams):
        """
        Add target columns metadata
        Args:
            outputs_metadata: metadata.base.DataMetadata
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            List[OrderedDict]
        """
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_name = "output_{}".format(column_index)
            column_metadata = OrderedDict()
            semantic_types = set()
            semantic_types.add(hyperparams["return_semantic_type"])
            column_metadata['semantic_types'] = list(semantic_types)

            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

    def _write(self, inputs:Inputs):
        inputs.to_csv(str(time.time())+'.csv')

    # def _get_sub_sequences_length(self, n_samples, window_size, step):
    #     """Pseudo chop a univariate time series into sub sequences. Return valid
    #     length only.
    #     Parameters
    #     ----------
    #     X : numpy array of shape (n_samples,)
    #         The input samples.
    #     window_size : int
    #         The moving window size.
    #     step_size : int, optional (default=1)
    #         The displacement for moving window.
    #     Returns
    #     -------
    #     valid_len : int
    #         The number of subsequences.

    #     """
    #     valid_len = int(np.floor((n_samples - window_size) / step)) + 1
    #     return valid_len



# def extract_features(X, n_samples, n_windows, indices):
#     print("X:", X.shape)
#     X_new = np.empty((n_samples, 3 * n_windows))
#     for j in range(n_windows):
#         start, end = indices[j]
#         arange = np.arange((start - end + 1) / 2, (end + 1 - start) / 2)
#         if end - start == 1:
#             var_arange = 1.
#         else:
#             var_arange = np.sum(arange ** 2)

#         for i in range(n_samples):
#             mean = np.mean(X[i, start:end])
#             X_new[i, 3 * j] = mean
#             X_new[i, 3 * j + 1] = np.std(X[i, start:end])
#             X_new[i, 3 * j + 2] = (np.sum((X[i, start:end] - mean) * arange) / var_arange)

#     print("X_new:", X_new.shape)
#     return X_new   

def extract_features(X, n_samples, n_windows, indices):
    print("X:", X.shape)
    X_new = np.empty((n_samples, 3 * n_windows))
    for j in range(n_windows):
        start, end = indices[j]
        arange = np.arange((start - end + 1) / 2, (end + 1 - start) / 2)
        if end - start == 1:
            var_arange = 1.
        else:
            var_arange = np.sum(arange ** 2)

        for i in range(n_samples):
            mean = np.mean(X[i, start:end])
            X_new[i, 3 * j] = mean
            X_new[i, 3 * j + 1] = np.std(X[i, start:end])
            X_new[i, 3 * j + 2] = (np.sum((X[i, start:end] - mean) * arange) / var_arange)

    print("X_new:", X_new.shape)
    return X_new.flatten(order = 'F')


class WindowFeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor over a window.
    This transformer extracts 3 features from each window: the mean, the
    standard deviation and the slope.
    Parameters
    ----------
    n_windows : int or float (default = 1.)
        The number of windows from which features are extracted.
    min_window_size : int or float (default = 1)
        The minimum length of the windows. If float, it represents a percentage
        of the size of each time series.
    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.
    Attributes
    ----------
    indices_ : array, shape = (n_windows, 2)
        The indices for the windows.
        The first column consists of the starting indices (included)
        of the windows. The second column consists of the ending indices
        (excluded) of the windows.
    """

    def __init__(self, n_windows=1., min_window_size=1, random_state=None):
        self.n_windows = n_windows
        self.min_window_size = min_window_size
        self.random_state = random_state
        self.time = str(time.time())

    def fit(self, X, y=None):
        """Fit the model according to the given training data.
        It generates the indices of the windows from which the features will be
        extracted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.
        y
            Ignored
        Returns
        -------
        self : object
        """
        X_crt = X[0]
        for x in X:
            if x.shape[1] < X_crt.shape[1]:
                X_crt = x
        X_crt = check_array(X_crt, dtype='float64')
        n_timestamps = X_crt.shape[1]
        n_windows, min_window_size, rng = self._check_params(X_crt)

        # print("111111111")

        # Generate the start and end indices
        start = rng.randint(0, n_timestamps - min_window_size, size=n_windows)
        # print("222222222")
        end = rng.randint(start + min_window_size, n_timestamps + 1, size=n_windows)
        # print("3333333")
        self.indices_ = np.c_[start, end]


        return self

    def transform(self, X):
        """Transform the provided data.
        It extracts the three features from all the selected windows
        for all the samples.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.
        Returns
        -------
        X_new : array, shape = (n_samples, 3 * n_windows)
            Extracted features.
        """
        # X = X.T
        # np.empty((n_samples, 3 * n_windows))
        # X_feature = None

        X[0] = check_array(X[0], dtype='float64')
        n_samples = X[0].shape[0]
        n_windows = self.indices_.shape[0]
        X_feature = np.array([extract_features(X[0], n_samples, n_windows, self.indices_)])

        for i in range(1, len(X)):
            X[i] = check_array(X[i], dtype='float64')
            # check_is_fitted(self)
            # Extract the features from each window
            
            print("i:", i)
            features =  np.array([extract_features(X[i], n_samples, n_windows, self.indices_)])
            X_feature = np.concatenate((X_feature, features), axis=0)

        return X_feature
        

    def _check_params(self, X):
        n_samples, n_timestamps = X.shape

        if not isinstance(self.n_windows,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'n_windows' must be an integer or a float.")
        if isinstance(self.n_windows, (int, np.integer)):
            if self.n_windows < 1:
                raise ValueError(
                    "If 'n_windows' is an integer, it must be positive "
                    "(got {0}).".format(self.n_windows)
                )
            n_windows = self.n_windows
        else:
            if self.n_windows <= 0:
                raise ValueError(
                    "If 'n_windows' is a float, it must be greater "
                    "than 0 (got {0}).".format(self.n_windows)
                )
            n_windows = ceil(self.n_windows * n_timestamps)

        if not isinstance(self.min_window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'min_window_size' must be an integer or a float.")
        if isinstance(self.min_window_size, (int, np.integer)):
            if not 1 <= self.min_window_size <= n_timestamps:
                raise ValueError(
                    "If 'min_window_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.min_window_size)
                )
            min_window_size = self.min_window_size
        else:
            if not 0 < self.min_window_size <= 1:
                raise ValueError(
                    "If 'min_window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {}).".
                    format(self.min_window_size)
                )
            min_window_size = ceil(self.min_window_size * n_timestamps)

        rng = check_random_state(self.random_state)

        return n_windows, min_window_size, rng

