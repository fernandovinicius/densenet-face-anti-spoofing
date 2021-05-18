from utils import load_model
from utils import single_predict
from utils import save_models_metrics
from PIL import Image
import pandas as pd
import argparse
import time
import os


def validation_routine(model_path, model_name, dataset_dir, output_dir):

    print(' ' + '-'*50 + '\n')
    print(f' >> {model_name} Validation \n')
    print(f'    Loading model: "{model_path}"...', end='')

    # Load tuned model
    model = load_model(model_path=model_path)
    print('Ok!')

    # Predicting all images in 'dataset_dir'
    labels = ['real', 'spoof']

    pred_info = []
    for label_int, label_str in enumerate(labels):

        p = os.path.join(dataset_dir, label_str)
        filelist = os.listdir(p)

        for cnt, file in enumerate(filelist):

            imgpath = os.path.join(p, file)
            img = Image.open(imgpath)

            t_start = time.time()
            pred, trust = single_predict(model, img)
            t_elapsed = 1000 * (time.time() - t_start)

            print(f'   {label_str}-{cnt}:  pred:{pred},   trust:{trust:.3f}'
                  f',   elapsed: {t_elapsed:.1f} ms')

            pred_info.append([file, label_int, pred, trust, t_elapsed])

    # Check if 'output_dir' exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save images predictions info
    info_filepath = os.path.join(output_dir,
                                 model_name + '_test_metrics.csv')
    df = pd.DataFrame(
        data=pred_info,
        columns=['file', 'label', 'pred', 'trust', 'time_ms']
    )
    df.to_csv(info_filepath, sep=';', index=False)

    # Save metrics
    metrics_filepath = os.path.join(output_dir,
                                    model_name + '_test_metrics.txt')
    save_models_metrics(
        model_name=model_name,
        metrics_filepath=metrics_filepath,
        y_true=df['label'].values,
        y_pred=df['pred'].values,
        t_pred=df['time_ms'].values,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        help='Model path.',
                        type=str)
    parser.add_argument('--model-name',
                        help='Model name.',
                        type=str)
    parser.add_argument('--dataset-dir',
                        help='Validation dataset directory.' +
                             ' Must contain "real" and "spoof" subdirs.',
                        type=str)
    parser.add_argument('--output-dir',
                        help='Dir where metrics will be stored.',
                        default='./',
                        type=str)

    args = parser.parse_args()
    print(args)
    validation_routine(
        model_path=args.model_path,
        model_name=args.model_name,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
