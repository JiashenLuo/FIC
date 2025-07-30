import argparse

params_float = ["cl_lambda", "noise_magnitude", "temperature", "temp","alpha","beta"]
params_int = ["neg_c", "path_num", "path_num1", "test_epoch", "layer_num", "sep_c", "p",
              "M"]

params_type_map = dict(zip(params_float, [float]*len(params_float)))
params_type_map.update(dict(zip(params_int, [int]*len(params_int))))


def parse_args():
    """return data_name, model_name, params"""
    parser = argparse.ArgumentParser(description='Process some integers.')
    # 添加一个参数，该参数可以接收多个键值对，格式为key=value
    parser.add_argument("--data_name", type=str, help="Name of the dataset")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument('--params', nargs='*', type=str, help='Dictionary in the form of key=value pairs')
    args = parser.parse_args()
    data_name, model_name = args.data_name, args.model_name
    params = {}
    if args.params:
        invalid_params = []
        for pair in args.params:
            key, value = pair.split('=')
            if key in params_type_map:
                params[key] = params_type_map[key](value)
            else:
                invalid_params.append(key)
        if invalid_params:
            raise ValueError(f"Invalid parameter: {invalid_params} 这些参数没有定义数据类型")
    return data_name, model_name, params

if __name__ == '__main__':
    # python parse.py --data_name=mnist --model_name=mlp --params=cl_lambda=0.1
    data_name, model_name, params = parse_args()
    print(data_name, model_name, params)
    if params:
        print(params)