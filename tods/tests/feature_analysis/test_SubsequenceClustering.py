import unittest
import os
import numpy as np

from d3m import container, utils
from d3m.metadata import base as metadata_base
from tods.feature_analysis import RandomSubsequenceSegmentation


class RandomSubsequenceSegmentationTest(unittest.TestCase):
    def test_basic(self):
        self.maxDiff = None
        # main = container.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.], 'c': [3., 4., 5.],},
        #                             # columns=['a', 'b', 'c'],
        #                             generate_metadata=True)

        this_path = os.path.dirname(os.path.abspath(__file__))
        table_path = os.path.join(this_path, '../../../datasets/data.npy')
        main = np.load(table_path, allow_pickle=True)
        print(main)


        # self.assertEqual(utils.to_json_structure(main.metadata.to_internal_simple_structure()), [{
        #     'selector': [],
        #     'metadata': {
        #         # 'top_level': 'main',
        #         'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
        #         'structural_type': 'd3m.container.pandas.DataFrame',
        #         'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
        #         'dimension': {
        #             'name': 'rows',
        #             'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
        #             'length': 3,
        #         },
        #     },
        # }, {
        #     'selector': ['__ALL_ELEMENTS__'],
        #     'metadata': {
        #         'dimension': {
        #             'name': 'columns',
        #             'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
        #             'length': 3,
        #         },
        #     },
        # }, {
        #     'selector': ['__ALL_ELEMENTS__', 0],
        #     'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        # }, {
        #     'selector': ['__ALL_ELEMENTS__', 1],
        #     'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        # }, {
        #     'selector': ['__ALL_ELEMENTS__', 2],
        #     'metadata': {'structural_type': 'numpy.float64', 'name': 'c'}
        # }])


        # self.assertIsInstance(main, container.DataFrame)


        hyperparams_class = RandomSubsequenceSegmentation.RandomSubsequenceSegmentation.metadata.get_hyperparams()
        primitive = RandomSubsequenceSegmentation.RandomSubsequenceSegmentation(hyperparams=hyperparams_class.defaults())

        primitive.set_training_data(inputs=main)
        primitive.fit()
        new_main = primitive.produce(inputs=main).value
        print(new_main.shape) 


        # self.assertEqual(utils.to_json_structure(main.metadata.to_internal_simple_structure()), [{
        #     'selector': [],
        #     'metadata': {
        #         # 'top_level': 'main',
        #         'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
        #         'structural_type': 'd3m.container.pandas.DataFrame',
        #         'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
        #         'dimension': {
        #             'name': 'rows',
        #             'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
        #             'length': 3,
        #         },
        #     },
        # }, {
        #     'selector': ['__ALL_ELEMENTS__'],
        #     'metadata': {
        #         'dimension': {
        #             'name': 'columns',
        #             'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
        #             'length': 3,
        #         },
        #     },
        # }, {
        #     'selector': ['__ALL_ELEMENTS__', 0],
        #     'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        # }, {
        #     'selector': ['__ALL_ELEMENTS__', 1],
        #     'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        # }, {
        #     'selector': ['__ALL_ELEMENTS__', 2],
        #     'metadata': {'structural_type': 'numpy.float64', 'name': 'c'}
        # }])



if __name__ == '__main__':
    unittest.main()
