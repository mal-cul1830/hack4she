import Swiper from 'https://unpkg.com/swiper/swiper-bundle.esm.browser.min.js'
var mySwiper = new Swiper(".swiper-container", {
    direction: "vertical",
    loop: true,
    pagination: ".swiper-pagination",
    grabCursor: true,
    speed: 1000,
    paginationClickable: true,
    parallax: true,
    autoplay: true,
    effect: "slide",
    keyboard: {
        enabled: true,
        onlyInViewport: false,
      },
  });
  

