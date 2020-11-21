const getnews = async () => {
    var url = 'http://newsapi.org/v2/everything?' +
    'q=breast cancer&' +
    'from=2020-11-20&' +
    'sortBy=relevance&' +
    'apiKey=f72bb86771a6462a958ddedd1523097a';
    const response = await fetch(url);
    const myJson = await response.json(); //extract JSON from the http response
    // do something with myJson
    console.log(myJson);
    articles = myJson.articles;
    for(i=0;i<7;++i){
        create_news(articles[i].title,articles[i].url,articles[i].description,articles[i].urlToImage);
    }
  }

  
var a = ["one", "two", "three", "four", "five", "six", "seven"];
let x = 0;

function create_news(ttext, link, content, imgurl){
    news = document.createElement('div');
    news.className = 'swiper-slide swiper-slide-'+a[x++];

    leftimg = document.createElement('div');
    leftimg.className = 'swiper-image';
    leftimg.setAttribute('data-swiper-parallax-', '-20%');
    leftin = document.createElement('div');
    leftin.className = 'swiper-image-inner swiper-image-left';
    title = document.createElement('h1');
    titlespan = document.createElement('span');
    titlespan.textContent = ttext;
    pleft = document.createElement('p');
    pleft.innerHTML = '<a href = "'+link+'">Go to article</a>';
    leftin.style.backgroundImage = "url('"+imgurl+"')";
    title.appendChild(titlespan);
    leftin.appendChild(title);
    leftin.appendChild(pleft);
    leftimg.appendChild(leftin);
    console.log(leftimg);

    rightimg = document.createElement('div');
    rightimg.className = 'swiper-image';
    rightimg.setAttribute('data-swiper-parallax-', '35%');
    rightin = document.createElement('div');
    rightin.className = 'swiper-image-inner swiper-image-right';
    rightin.style.backgroundColor = 'black';
    rightin.style
    pright = document.createElement('p');
    pright.innerText = content;
    rightin.appendChild(pright);
    rightimg.appendChild(rightin);

    news.appendChild(leftimg);
    news.appendChild(rightimg);

    e = document.getElementById('news');
    e.append(news);
    console.log(e);

}
getnews();
