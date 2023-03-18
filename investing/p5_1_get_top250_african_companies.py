# https://african.business/dossiers/africas-top-companies/

# "https://datawrapper.dwcdn.net/t4748/1/"

import pickle

from bs4 import BeautifulSoup
import requests


def save_top250_africa_tickers():
    """

    :return:
    """
    headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    response = requests.get('https://african.business/dossiers/africas-top-companies/', headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    """
    You need to find the attr. 'data-src' within the tag iframe
    <div class="wp-block-embed__wrapper">
       <iframe aria-label="Table" class="lazyload" data-src="https://datawrapper.dwcdn.net/t4748/1/" frameborder="0" height="5851" id="datawrapper-chart-t4748" scrolling="no" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" style="width: 0; min-width: 100% !important; border: none;" title="Africa's Top 250 Companies in 2022">
       </iframe>
       <script type="text/javascript">
        !function(){"use strict";window.addEventListener("message",(function(e){if(void 0!==e.data["datawrapper-height"]){var t=document.querySelectorAll("iframe");for(var a in e.data["datawrapper-height"])for(var r=0;r<t.length;r++){if(t[r].contentWindow===e.source)t[r].style.height=e.data["datawrapper-height"][a]+"px"}}}))}();
       </script>
      </div>
    """

    # print(soup.prettify())

    iframes = soup.find_all('iframe')
    # print(iframes)
    for iframe in iframes:
        src = iframe['data-src']
        print(src)
        response = requests.get(src, headers=headers)
        data_soup = BeautifulSoup(response.text, 'lxml')
        print(data_soup.prettify())

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    response = requests.get('https://datawrapper.dwcdn.net/t4748/1/', headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    print(soup.prettify())

    tickers = list()


if __name__ == '__main__':
    save_top250_africa_tickers()

