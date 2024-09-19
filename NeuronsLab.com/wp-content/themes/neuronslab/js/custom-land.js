
$(function(){
    
    $(document).ready(function() {
      if($('.js-slider-services').length) {
          var contServices = new Splide( '#content-services', {
            type: 'fade',
            rewind: true,
            autoplay: true,
            pagination: false,
            arrows: false,
            cover: true,
            autoplay: true,
            interval: 5000
          });
          
          var navServices = new Splide( '#logo-services', {
            rewind: false,
            isNavigation: true,
            arrows: false,
            pagination: false,
            perPage: 3,
            drag: false,
            autoplay: true,
            interval: 5000,
            gap:10,
            breakpoints: {
                 '1024': {
                   perPage: 3,
                 },
                 '767': {
                   perPage: 1,
                    arrows: true,                    
                 }
            }
          });
          
          contServices.sync( navServices );
          contServices.mount();
          navServices.mount();

      }

    });
    
    $(".box-land-how__item:first-child .box-land-how__info").css({'display':'block'})
    $(".box-land-how__title").on('click', function(e) {
        e.preventDefault();
        $(this).parent().siblings().find(".box-land-how__info").slideUp();
        $(this).next(".box-land-how__info").slideToggle();
    });
    
    
    $(".acord-info__name").click(function(e) {
        e.preventDefault();
        var $item = $(this).parent(".acord-info__item");
        var $content = $item.children(".acord-info__cont");
        $(".acord-info__item").not($item).removeClass("open").children(".acord-info__cont").slideUp();
        $item.toggleClass("open");
        $content.slideToggle();
    });
    
    $(".list-capabilities__btn-open").click(function(e) {
        e.preventDefault();
        var $item = $(this).parent(".list-capabilities__item");
        $(this).parent(".list-capabilities__item").find(".list-capabilities__open-text").slideToggle();
        $(".list-capabilities__item").not($item).removeClass("open").find(".list-capabilities__open-text").slideUp();
        $item.toggleClass("open");
    
    });
    
    if ($(window).width() < 768) {
        if ($('.js-slider-clude').length) {
            var splide4 = new Splide('.js-slider-clude', {
                type: 'slide',
                arrows: 'auto',
                pagination: false,
                perPage: 1,                
                autoScroll: {
                    speed: 1,
                },
                gap: '0rem',
                rewind: false
            });

            splide4.on( 'mounted', function () {
                var currentSlide = splide4.index + 1;
                $('.slider-cloud__num-current').text(currentSlide);
                $('.slider-cloud__num-total').text(splide4.Components.Slides.getLength());
            });
            splide4.mount();

            var slideCount4 = splide4.length;
            var visibleSlides4 = splide4.options.perPage;

            if (slideCount4 <= visibleSlides4) {
                $('.js-slider-clude .splide__arrows').addClass('arr-hide');
            } else {
                $('.js-slider-clude .splide__arrows').removeClass('arr-hide');
            }

            splide4.on('move', function (newIndex, prevIndex, destIndex) {
                var currentSlide = splide4.index + 1;
                $('.slider-cloud__num-current').text(currentSlide);
            });
     

        }
    } else {
        if ($('.js-slider-clude').length) {
            $('.js-slider-clude').each(function () {
                $(this).data('splide').destroy();
            });
        }
    }

    // sprint overview slider

    if ($(".sprint-overview__slider").length) {
        var splide = new Splide(".sprint-overview__slider", {
            pagination: false,
            gap: "0.7rem",
            perPage: 3,
            // autoplay: true,
            // interval: 3000,
            updateOnMove: true,
            focus: 0,
            drag: false,
            perMove: 1,
            // isNavigation: true,
            breakpoints: {
                '1024': {
                  perPage: 4,
                    drag: false,
                },
                '767': {
                    perPage: 2,
                },
           }
        });
        splide.on('ready', function () {
            setInterval(() => {
                splide.go('>');
            }, 3000);
        });
        splide.mount();
        splide.on( 'moved', function () {
            if (splide.Components.Controller.getNext() === -1) {
                setTimeout(() => {
                    splide.go(0);
                }, 3000);
            }
        });
    }

    if($('.js-title-slider').length) {
        var splideText = new Splide(".js-title-slider", {
            type: "loop",
            direction: "ttb",
            heightRatio: 0.7,
            arrows: false,
            pagination: false,
            speed: 1000,
            autoplay: true,
            interval: 3000,
            drag: false,
        });
        splideText.mount();
    }
});





