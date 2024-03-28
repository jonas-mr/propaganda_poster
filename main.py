import argparse
import logging
import pickle
import sys




def usage():
    return "Usage: %s [options]" % sys.argv[0]
def main():
    #logger = logging.getLogger('propagandaPoster')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='''An application to scrape, process and visualize any kind of image collection.''')
    parser.add_argument('files', metavar='file', type=str, default='./datasets/propagandaSet')
    parser.add_argument('-s', '--scrape', action='store_true', help='', default=False)
    parser.add_argument('-e', '--experiment', action='store_true', help='', default=False)
    parser.add_argument('-v', '--visualize', action='store_true', help='', default=False)
    parser.add_argument('-m', '--model', help='', type=str, default='vgg16')
    parser.add_argument('-c', '--components', help='',type=int, default=100)

    args = parser.parse_args()
    data_root = args.files
    scraping =args.scrape
    experiment = args.experiment
    model = args.model + '_'+ str(args.components)
    visualize = args.visualize
    components = args.components


    if scraping:
        import scraper
        scraper.DataScraper(data_root)

    elif experiment:
        import construct_dataset
        import data_experiment
        import preprocessor
        dataset_constructor = construct_dataset.ConstructDataset(data_root)
        preprocessor.Preprocessor(data_root, dataset_constructor.dataset, model, components)
        data_experiment.DataExperiment(data_root, model)

    if visualize:
        import dash_app


if __name__ == "__main__":
    main()