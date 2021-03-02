import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.detection_algorithm.SystemWiseDetection import SystemWiseDetectionPrimitive

class SystemWiseDetectionTestCase(unittest.TestCase):
    def test_basic(self):

        main = container.DataFrame({'timestamp': [1,2,3,4,5,6,7,8,9,10,11,12], 'system_id': [1,1,1,1,5,5,5,5,3,3,3,3],'scores':[1.0,4.0,5.0,6.0,2.0,1.0,9.0,10.0,3.0,4.0,18.0,1.0]},columns=['timestamp', 'system_id', 'scores'],
                                   generate_metadata=True)




        self.assertEqual(utils.to_json_structure(main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 12,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'timestamp'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'system_id'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'scores'},
        }])

        hyperparams_class = SystemWiseDetectionPrimitive.metadata.get_hyperparams()
        hyperparams = hyperparams_class.defaults()
        print(hyperparams_class.__dict__)
        method_types = ['max', 'avg', 'sliding_window_sum','majority_voting_sliding_window_sum','majority_voting_sliding_window_max']
        for method_type in method_types:
            hyperparams = hyperparams.replace({'method_type': method_type})
            primitive = SystemWiseDetectionPrimitive(hyperparams=hyperparams)
            output_main = primitive.produce(inputs=main).value
            print(output_main)
            self.assertEqual(list(output_main.columns), [0, 1])



        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()

