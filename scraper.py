import glob
import json
from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO
import logging
logging.basicConfig(level=logging.INFO)
class DataScraper:

    def __init__(self, data_root):
        self.data_root = data_root
        self.url_collection = [
            'https://content.libraries.wsu.edu/iiif/2/propaganda/manifest.json',
            # 'https://digitalcollections.hclib.org/iiif/2/p17208coll3/manifest.json',
            'https://cdm16002.contentdm.oclc.org/iiif/2/p16002coll9/p1.json',
            'https://www.idaillinois.org/iiif/2/isl5/manifest.json',
            'https://hrc.contentdm.oclc.org/iiif/2/p15878coll26/manifest.json',
            'https://collections.digitalmaryland.org/iiif/2/mdwp/manifest.json',
        ]
        for url in self.url_collection:
            logging.info(f'Scraping from {url}')
            if not self.url_already_visited(url):
                self.contentdm_api(url)
                self.url_visit_finished(url)

        self.chinese_poster()
        for i in range(1, 103):
            url = f'https://propaganda.pictures/page/{i}'
            if not self.url_already_visited(url):
                self.propaganda_archive(url)
                self.url_visit_finished(url)


    def write_metadata(self, image_path, image_metadata):
        metadata_json = json.dumps(image_metadata, indent=4, ensure_ascii=False)
        with open(f'{self.data_root}{image_path}.json', 'w', encoding='UTF-8') as outfile:
            outfile.write(metadata_json)

    def image_exists_already(self, img_id):
        all_files = [x.lstrip('dataotherchina\\/') for x in glob.glob(f'{self.data_root}data/*/*')]
        t = f'{img_id}.png' in all_files and f'{img_id}.json' in all_files
        return t

    @staticmethod
    def url_already_visited(url):
        with open('tmp/visited_urls.txt', 'r', encoding='UTF8') as file:
            file.seek(0)
            urls = file.read().splitlines()
            if url not in urls:
                return False
            else:
                return True

    @staticmethod
    def url_visit_finished(url):
        with open('tmp/visited_urls.txt', 'a') as file:
            file.write(url+'\n')

    def write_image(self, image_name, image_response):
        image_data = image_response.content
        image = Image.open(BytesIO(image_data))
        if image.mode in ("RGBA", "P"):
            logging.info('image is transparent oder so')
            image = image.convert('RGB')
        image.thumbnail((448,448), resample=Image.HAMMING)
        image.save(f'{self.data_root}{image_name}.png')



    def delete_redundant_files(self):
        all_files = [x.lstrip('other\\') for x in glob.glob(f'{self.data_root}data/*')]
        for file in all_files:
            if not (file.endswith('.jpg') or file.endswith('.json') or file.endswith('.txt')):
                import os
                logging.info(f'deleting {file}')
                os.remove(f'data/{file}')


    def contentdm_api(self,url):
        response = requests.get(url)
        data = response.json()
        for item in data['manifests']:
            if not self.url_already_visited(item['@id']):
                item_response = requests.get(item['@id'])

                if item_response.status_code == 200:
                    item_data = item_response.json()

                    image_metadata = {x['label']: x['value'] for x in item_data['metadata']}


                    image_name = image_metadata["Identifier"]
                    if ';' in image_name:
                        image_name = image_name.split(";")[0]

                    if not self.image_exists_already(image_name):
                        self.write_metadata(f'data/other/{image_name}', image_metadata)

                        if (len(item_data['sequences']) == 1
                                and len(item_data['sequences'][0]['canvases']) == 1
                                and len(item_data['sequences'][0]['canvases'][0]['images']) == 1):

                            image_url = item_data['sequences'][0]['canvases'][0]['images'][0]['resource']['@id']
                            image_response = requests.get(image_url, stream=True)
                            if image_response.status_code == 200:
                                self.write_image(f'data/other/{image_name}', image_response)
                                self.url_visit_finished(item['@id'])

                            logging.info(f"Image and Metadata for {image_name} written.")
                        else:
                            logging.warning(f"Sequences in {item['@id']} unusual large.")
                            count = 0
                            for image_data in item_data['sequences'][0]['canvases']:
                                count += 1
                                image_url = image_data['images'][0]['resource']['@id']
                                image_response = requests.get(image_url, stream=True)
                                if image_response.status_code == 200:
                                    self.write_image(f'data/other/{image_name}_{count}', image_response)
                                    self.url_visit_finished(image_url)
                                    logging.info(f"Image and Metadata for {image_name}_{count} written.")
                            self.url_visit_finished(item['@id'])



        return logging.info('Finished scraping contentdmAPIs')



    def chinese_poster(self):
        url = 'https://chineseposters.net/posters/posters?field_theme_target_id=All&field_tags_target_id=All&field_taxonomy_artists_target_id=All&field_collection_target_id=All&field_datesearch_target_id=All&sort_by=date&sort_order=ASC&items_per_page=All'
        page = requests.get(url)
        html = page.text
        soup = BeautifulSoup(html, 'html.parser')

        results = soup.findAll('a', class_=None)
        for result_link in results:
            if not result_link.find('img') and not self.url_already_visited(f"https://chineseposters.net/{result_link.get('href')}"):
                # Get Image- and Meta-other
                result_entry = requests.get(f"https://chineseposters.net/{result_link.get('href')}")
                if result_entry.status_code == 200:
                    html_entry = result_entry.text
                    soup_entry = BeautifulSoup(html_entry, 'html.parser')
                    main_block = soup_entry.find('div', class_='w3-container node__content')

                    # parse and save Metadata
                    poster_title = soup_entry.find('h1', class_='w3-margin-left w3-padding-24 w3-xxlarge page-title')
                    image_metadata = {'Title': poster_title.text,
                                      'Designer': '',
                                      'Date': '',
                                      'Publisher':'',
                                      'Size': '',
                                      'Call nr.':'',
                                      'Collection':'',
                                      'Notes':'',
                                      'Theme':'',
                                      'Tags':[]
                                      }
                    image_url = main_block.find('img', class_='w3-image').get('src')
                    image_metadata_list = main_block.text.split('\n')

                    last_key = ''
                    for line in image_metadata_list:
                        if line in image_metadata.keys():
                            last_key = line
                        elif last_key and line.strip() and type(image_metadata[last_key]) is str:
                            image_metadata[last_key] += line
                        elif last_key and line.strip() and type(image_metadata[last_key]) is list:
                            image_metadata[last_key].append(line)

                    img_id = result_link.get('href').split('/')[-1]

                    if not self.image_exists_already(img_id):
                        self.write_metadata(f'data/china/{img_id}', image_metadata)

                        # Get and Write image
                        image_response = requests.get(f"https://chineseposters.net/{image_url}", stream=True)
                        if image_response.status_code == 200:
                            self.write_image(f'data/china/{img_id}', image_response)

                            self.url_visit_finished(f"https://chineseposters.net/{result_link.get('href')}")
                            logging.info(f'Image and Metadata for {img_id} written.')
                    elif self.image_exists_already(img_id):
                        self.url_visit_finished(f"https://chineseposters.net/{result_link.get('href')}")
                        logging.info(f'Image and Metadata for {img_id} already exists.')

        return logging.info('Finished scraping chinesepropaganda')


    def propaganda_archive(self,url):
        page = requests.get(url)
        html = page.text
        soup = BeautifulSoup(html, 'html.parser')
        main_block = soup.find('main')
        results = main_block.findAll('li')

        for result_list_element in results:
            result_link = result_list_element.find('a').get('href')
            if not self.url_already_visited(result_link):
                result_entry = requests.get(result_link)
                if result_entry.status_code == 200:
                    html_entry = result_entry.text
                    soup_entry = BeautifulSoup(html_entry, 'html.parser')
                    main_block = soup_entry.find('main')

                    # Parse and save metadata
                    metadata_block = main_block.find('div', class_='wp-block-column is-layout-flow wp-block-column-is-layout-flow')

                    try:
                        additional_metadata_block = metadata_block.find('div', class_='wp-block-group is-layout-flex wp-container-9 wp-block-group-is-layout-flex').findAll('div')[1:]
                    except AttributeError:
                        additional_metadata_block = metadata_block.find('div', class_='wp-block-group is-layout-flex wp-container-11 wp-block-group-is-layout-flex').findAll('div')[1:]
                    if len(additional_metadata_block) <= 3:
                        image_metadata = {'post_author': additional_metadata_block[0].text,
                                      'upload_date': additional_metadata_block[1].text,
                                      'region': additional_metadata_block[2].text,
                                      'title': metadata_block.find('h2').text}
                    else:
                        image_metadata = {'post_author': additional_metadata_block[0].text,
                                          'upload_date': additional_metadata_block[1].text,
                                          'region': additional_metadata_block[2].text,
                                          'tags': additional_metadata_block[3].text.split(','),
                                          'title': metadata_block.find('h2').text}

                    ps = metadata_block.findAll('p', class_=None)
                    last_key = None
                    for p in ps:
                        if not p.find('strong') and last_key:
                            image_metadata[last_key]  = p.text
                            last_key = None
                        elif p.find('strong'):
                            image_metadata[p.find('strong').text] = ''
                            last_key  = p.find('strong').text
                            if len(p.text.split(':')) > 1:
                                image_metadata[p.find('strong').text] = ' '.join(p.text.split(':')[1:])

                    img_id = result_link.strip('https://propaganda.pictures/archives/')

                    if not self.image_exists_already(img_id):
                        self.write_metadata(f'data/other/{img_id}', image_metadata)

                        # Get and write image
                        img_link = main_block.find('img').get('src')
                        image_response = requests.get(img_link, stream=True)
                        if image_response.status_code == 200:
                            self.write_image(f'data/other/{img_id}', image_response)

                            self.url_visit_finished(result_link)
                            logging.info(f"Image and Metadata for {img_id} written.")
                    else:
                        self.url_visit_finished(result_link)

        return logging.info('Finished scraping chinesepropaganda')




    '''
    def hoover_collection():
        url = 'https://digitalcollections.hoover.org/groups/formats/category/Poster?filter=approved%3Atrue#filters'
        page = requests.get(url)
        html = page.text
        soup = BeautifulSoup(html, 'html.parser')
        collection_block = soup.find('div', id='tlistview')
        results = collection_block.findAll('div', class_='item list-item col-lg-12 col-md-12 col-sm-12 col-12')
    
        for result_list_element in results:
            result_link = result_list_element.find('a')
            result_entry = requests.get(result_link.get('href'))
            html_entry = result_entry.text
            soup_entry = BeautifulSoup(html_entry, 'html.parser')
            main_block = soup_entry.find('main')
    
            # Parse and save metadata
            metadata_block = main_block.find('div', class_='wp-block-column is-layout-flow wp-block-column-is-layout-flow')
    
            additional_metadata_block = metadata_block.find('div', class_='wp-block-group is-layout-flex wp-container-9 wp-block-group-is-layout-flex').findAll('div')[1:]
            image_metadata = {'post_author': additional_metadata_block[0].text,
                              'upload_date': additional_metadata_block[1].text,
                              'region': additional_metadata_block[2].text,
                              'tags': additional_metadata_block[3].text.split(','),
                              'title': metadata_block.find('h2').text}
            ps = metadata_block.findAll('p', class_=None)
            for p in ps:
                image_metadata[p.find('strong').text] = p.text.split(':')[1].strip(' .')
    
            img_id = result_link.get('href').strip('https://propaganda.pictures/archives/')
            metadata_json = json.dumps(image_metadata, indent=4, ensure_ascii=False)
            with open(f'other/{img_id}.json', 'w', encoding='UTF-8') as outfile:
                outfile.write(metadata_json)
                
    '''


