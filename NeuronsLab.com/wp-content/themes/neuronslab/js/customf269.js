/* viewport width */
function viewport(){
	var e = window, 
		a = 'inner';
	if ( !( 'innerWidth' in window ) )
	{
		a = 'client';
		e = document.documentElement || document.body;
	}
	return { width : e[ a+'Width' ] , height : e[ a+'Height' ] }
};
/* viewport width */
//$(window).on('load', function () {
	if( /Android|webOS|iPhone|iPad|iPod|BlackBerry/i.test(navigator.userAgent) ) {
		$('body').addClass('ios');
	} else{
		$('body').addClass('web');
	};
    setTimeout(function () {
        $(".js-bg").each(function () {
            $(this).css('background-image', 'url(' + $(this).data("preload") + ')');
        });
        $("[data-src]").each(function () {
            $(this).attr('src', $(this).data("src"));
        });
    }, 200);
    
    var viewport_wid = viewport().width;
	var viewport_height = viewport().height;
	
	if (viewport_wid <= 1024) {
		$('.has-drop > a').on('click', function() {
          $(this).parent().find('.drop-menu').slideToggle();
            $(this).parent().toggleClass('active');
            return false;
        });
	} else{
        $('.has-drop > a').on('click', function() {        
            return false;
        });
    }
    
    if($('.inf-evol__imgs').length) {

      const chunkArray = (arr, size) => arr.length > size ? [arr.slice(0, size), ...chunkArray(arr.slice(size), size)] : [arr];
      const imgsArr = chunkArray($('.inf-evol__imgs img'), 50);

      $(imgsArr).each(function(i, item) {

        if (i == 0) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 2000);
        } else if (i == 1) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 3000);
        } else if (i == 2) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 4000);
        } else if (i == 3) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 5000);
        } else if (i == 4) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 6000);
        }
      });
    }
//});




$(function(){
    var viewport_height = viewport().height;
    $('.video-about-bg').height(viewport_height+65);
    
    setTimeout(function () {
	   $('body').removeClass('loaded'); 
    }, 100);
    
    
    /*$('.stories-carusel__item img').on('click', function(e) {
        e.preventDefault();
      $(this).parent().toggleClass('opened');
      $(this).parent().siblings().removeClass('opened');
    });
    
    
    $('.stories-carusel__item .stories-carusel__descr').on('touchstart', function(e) {
       
      $(this).parent().toggleClass('opened');
      $(this).parent().siblings().removeClass('opened');
    });
    $('.stories-carusel__item img').on('touchstart', function(e) {
        e.preventDefault();
      $(this).parent().toggleClass('opened');
      $(this).parent().siblings().removeClass('opened');
    });*/
    
  var isSafari =
  /Safari/.test(navigator.userAgent) &&
  /Apple Computer/.test(navigator.vendor);
  if (isSafari) {
    $("body").addClass("safari");
  }

	/* placeholder*/	   
	$('input, textarea').each(function(){
 		var placeholder = $(this).attr('placeholder');
 		$(this).focus(function(){ $(this).attr('placeholder', '');});
 		$(this).focusout(function(){			 
 			$(this).attr('placeholder', placeholder);  			
 		});
 	});
	/* placeholder*/

	$('.button-nav').click(function(){
		$(this).toggleClass('active'), 
		$('.main-nav-list').slideToggle(); 
		return false;
	});
	
    
    $(document).ready(function() {
      if($('.js-slider-logo').length) {
          var commentsLogo = new Splide( '#slider-comments', {
            type: 'fade',
            rewind: true,
            autoplay: true,
            pagination: false,
            arrows: false,
            cover: true,
            autoplay: true,
            interval: 5000
          });
          
          var navLogo = new Splide( '#logo-slider', {
            rewind: true,
            isNavigation: true,
            arrows: false,
            pagination: false,
            perPage: 5,
            drag: false,
            autoplay: true,
            interval: 5000,
            breakpoints: {
                 '1024': {
                   perPage: 4,
                 },
                 '768': {
                   perPage: 1,
                 }
            }
          });
          
          commentsLogo.sync( navLogo );
          commentsLogo.mount();
          navLogo.mount();

      }

    });
    
    
    $(document).ready(function() {
      if($('.info-video').length) {
          var main = new Splide( '#main-slider', {
            type: 'fade',
            rewind: true,
            autoplay: true,
            heightRatio: 0.5,
            pagination: false,
            pauseOnHover: false,
            arrows: false,
            interval: 6000,
            cover: true
          });
          
          var thumbnails = new Splide( '#thumbnail-slider', {
            rewind: true,
            isNavigation: true,
            arrows: false,
            pagination: false,
            pauseOnHover: false,
            interval: 6000,
            perPage: 3,
            drag: false
          });

          main.on('move', function (newIndex, prevIndex, destIndex) {
            main.Components.Elements.slides[newIndex].querySelector('.video-element').play();
            
            var currentSlide = main.Components.Elements.slides[main.index];
            var video = currentSlide.querySelector('video');
            if (video) {              
              var videoDuration = video.duration * 1000;
              main.options.interval = videoDuration;
              main.refresh();
            }
          });
          main.on('mounted', function () {
            var currentSlide = main.Components.Elements.slides[main.index];
            var video = currentSlide.querySelector('video');
            if (video) {              
              var videoDuration = video.duration * 1000;
              main.options.interval = videoDuration;
              main.refresh();
            }
          });
          
          main.sync( thumbnails );
          main.mount();
          thumbnails.mount();

          var $firstVideo = $('.video-slider .splide__slide:first-child .video-element');
          $firstVideo.get(0).play();
      }

    });
    
    if($('.js-carusel').length) {
      var splide = new Splide( '.js-carusel', {
        type: 'loop',
        //focus: 'center',
        arrows: true,
        pagination: false,
        perPage: 3,
        gap: '1.3rem',
        breakpoints: {
            '1024': {
                perPage: 2,
                gap: '1rem',
                focus: 'center'
            },
            '767': {
                perPage: 1,
                gap: '2rem',
                focus: 'center'
            }
        }
      });

      splide.on( 'mounted active', function () {
        var end  = splide.Components.Controller.getEnd() + 1;
        var rate = Math.min( ( splide.index + 1 ) / end, 1 );

        splide.root.querySelector( '.my-slider-progress-bar' ).style.width = String( 100 * rate ) + '%';
      });
   
      splide.mount();
        

    };
    $('.stories__title-link').on('click', function() {
        $(this).parents('section').find('.splide__arrow--next').click();
    });
    
    if($('.list-partners').length) {
        setInterval(function() {
          var listItems = $('.list-partners li');
          var activeItem = listItems.filter('.active');
          var nextItem = activeItem.next();
          if (nextItem.length === 0) {
            nextItem = listItems.first();
          }
          activeItem.removeClass('active');
          nextItem.addClass('active');
        }, 2000);
    };
    
//tabs    
    var timer;
    $(document).ready(function() {
      $('.js-tab-content').removeClass('open');
      $('.js-tab-content:first').addClass('open');

      $('ul.js-tabs li a').click(function(e) {
        e.preventDefault();

        var tab_id = $(this).parent().attr('data-tab');

        $('ul.js-tabs li').removeClass('active');
        $('.js-tab-content').removeClass('open');

        $(this).parent().addClass('active');
        $('#' + tab_id).addClass('open');
      });

      // При клике на контент таба меняем активную вкладку
      $('.js-tab-content').click(function() {
        var tab_id = $(this).attr('id');
        var active_tab = $('ul.js-tabs li.active').attr('data-tab');

        $('ul.js-tabs li').removeClass('active');
        $('.js-tab-content').removeClass('open');

        if (tab_id !== active_tab) {
          $('ul.js-tabs li[data-tab="' + tab_id + '"]').addClass('active');
          $('#' + tab_id).addClass('open');
        } else {
          $('ul.js-tabs li:first').addClass('active');
          $('.js-tab-content:first').addClass('open');
        }
      });

      $('ul.js-tabs, .js-tab-content').on('click', function() {
        clearTimeout(timer);
        timer = setTimeout(function() {
          var active_tab = $('ul.js-tabs li.active');
          var next_tab = active_tab.next('li');
          if (next_tab.length === 0) {
            next_tab = $('ul.js-tabs li:first');
          }
          next_tab.find('a').click();
        }, 9000);
      });
      timer = setTimeout(function() {
        $('ul.js-tabs li:first a').click();
      }, 9000);
    });
    
    
    
    $(document).ready(function() {
      // Обработчик события скролла
      $(window).scroll(function() {
        var scrollPosition = $(window).scrollTop();

        $('.js-color:visible').each(function(idx, item) {
          var sectionOffset = $(this).offset().top;
          var sectionHeight = $(this).outerHeight();

          if (scrollPosition >= sectionOffset && scrollPosition < sectionOffset + sectionHeight) {
            var sectionColorClass = $(item).attr("data-color");

            $('body').removeClass('grey-bg white-bg black-bg');

            $('body').addClass( sectionColorClass + '-bg');

            return false;
          }
        });
      });
    });
    

    
    $(document).ready(function() {

      if ($(window).width() > 767) {
        var $text = $('.text');
        var $videos = $('.video');
  
        $(window).scroll(function() {
          var scrollTop = $(this).scrollTop();
  
          $videos.each(function() {
            var videoTop = $(this).offset().top - 300;
            var videoHeight = $(this).height();
            var videoId = $(this).data('video-id');
  
            if (scrollTop >= videoTop && scrollTop < videoTop + videoHeight) {
              var playPromise = $(this).find('video')[0].play();
                $videos.not(this).find('video').each(function() {

                  if (playPromise !== undefined) {
                    playPromise.then(_ => {          
                      this.pause();
                      this.currentTime = 0;
                    })
                  }
      
              });
  
              $('.left-block__item').removeClass('active');
              $('.left-block__item').eq(videoId - 1).addClass('active');
            }
          });
        });
      }
    });
    
    if($('.js-carusel2').length) {
      var splide2 = new Splide( '.js-carusel2', {
        type: 'loop',
        focus: 'center',
        arrows: true,
        pagination: false,
        perPage: 4,
        gap: '1.3rem',
        breakpoints: {
            '1200': {
                perPage: 4,
                gap: '1rem',
                focus: 'center'
            },
            '1100': {
                perPage: 3,
                gap: '1rem',
                focus: 'center'
            },
            '767': {
                perPage: 2,
                gap: '1.3rem',
                focus: 'center'
            },
            '550': {
                perPage: 1,
                gap: '1.3rem',
                focus: 'center'
            }
        }
      });

      splide2.on( 'mounted move active', function (newIndex, prevIndex, destIndex) {

        var end  = splide2.Components.Controller.getEnd() + 1;
        var rate = Math.min( ( splide2.index + 1 ) / end, 1 );
        
        splide2.root.querySelector( '.my-slider-progress-bar' ).style.width = String( 100 * rate ) + '%';
      });
   
      splide2.mount();  
    };

    if($('.inf-evol__imgs').length) {

      const chunkArray = (arr, size) => arr.length > size ? [arr.slice(0, size), ...chunkArray(arr.slice(size), size)] : [arr];
      const imgsArr = chunkArray($('.inf-evol__imgs img'), 50);


      $(window).scroll(function() {
        var scrollTop = $(this).scrollTop();

        var imgsBlockHeight = $('.inf-evol__imgs').height();
        var imgsBlockTop = $('.inf-evol__imgs').offset().top - imgsBlockHeight / 2;
    

        if (scrollTop >= imgsBlockTop && scrollTop < imgsBlockTop + imgsBlockHeight) {
          let num1 = imgsBlockHeight / $('.inf-evol__imgs img').length;

				  let num = Math.round((scrollTop - imgsBlockTop) / num1);

          if (num <= $('.inf-evol__imgs img').length - 1) {
            $('.inf-evol__imgs img').removeClass('active');
            $('.inf-evol__imgs img').eq(num).addClass('active');
          }
        }
      });
    }

    if($('.right-block__imgs').length) {
      $(window).scroll(function() {

        var scrollTop = $(this).scrollTop();
  
        $('.right-block__imgs').each(function() {
  
          var imgsBlockHeight = $(this).height();
          var imgsBlockTop = $(this).offset().top - imgsBlockHeight / 2;
  
          if (scrollTop >= imgsBlockTop && scrollTop < imgsBlockTop + imgsBlockHeight) {
            let num1 = imgsBlockHeight / $(this).find('img').length;
  
            let num = Math.round((scrollTop - imgsBlockTop) / num1);
  
            if (num <= $(this).find('img').length - 1) {
              $(this).find('img').removeClass('active');
              $(this).find('img').eq(num).addClass('active');
            }
          }
  
        });
  
      });
    }

    

    const articleVideos = document.querySelectorAll('.stories-carusel__video');

    if (articleVideos) {
      if ($(window).width() >= 1025) {
          articleVideos.forEach(video => {
            video.controls = false;
            var isVideoEnded2 = false;
            video.addEventListener("ended", function() {
              isVideoEnded2 = true;
              this.currentTime = 0;
            });
            video.parentNode.addEventListener("mouseover", function() {
              if (isVideoEnded2) {
                video.currentTime = 0;
                isVideoEnded2 = false;
              }
              video.play();
            });
            video.parentNode.addEventListener("mouseout", function() {
              video.pause();
            });
          });
      }
    }

    $(".js-btn-menu").on("click", function (e) {
      e.preventDefault();
      $(this).toggleClass("open");
      $(".menu").toggleClass("active");
      $("html").toggleClass("scroll-off");
    });
    $(".mask-menu").on("click", function (e) {
      e.preventDefault();
      $(".menu").removeClass("active");
      $("html").removeClass("scroll-off");
      $(".js-btn-menu").removeClass("open");
    });
    
    
    if($('.js-carusel-vertical').length) {
      var splide3 = new Splide( '.js-carusel-vertical', {
        type: 'slide',
        direction: 'ttb',  
        heightRatio: 0.7, 
        focus: 'center',
        arrows: false,
        pagination: false,
        perPage: 1,
          gap: '2.5rem',
          loop: false,
        // autoScroll: {
        //   speed: 1,
        // },
        breakpoints: {
             '4000': {
               destroy: false,
             },
             '1024': {
               destroy: true,
             }
        }
      });
   
      splide3.mount();
    };
    
    if($('.js-slider-articles').length) {
      var splide5 = new Splide( '.js-slider-articles', {
        //focus: 'center',
        arrows: true,
        pagination: false,
        perPage: 1,
        autoplay: true,        
        interval: 6000,
        speed: 0,
        pauseOnHover: false,   
        loop: true,
        type:'loop'
      });
        
        $('.stories__title-link').on('click', function() {
            $(this).parents('.js-slider-articles').find('.splide__arrow--next').click();
            //$(this).parents('.js-slider-articles').find('.slick-next').click();
        });
        
        splide5.mount();

      var curNumElement = document.querySelector('.cur-num');
      splide5.on('moved', function (newIndex) {
        var currentSlideNumber = newIndex + 1;

        curNumElement.textContent = `${currentSlideNumber}`;
      });
        
        /*$('.js-slider-articles .splide__list').slick({
			infinite: true,
			speed: 500,
			slidesToShow: 1,
			slidesToScroll: 1,
            autoplay: true, 
            autoplaySpeed: 8000,
            fade:true,
            beforeChange: function (slick, currentSlide, nextSlide) {
                $('.cur-num').text(currentSlide + 1);
            }
		});*/
    };
    
    if($(".js-filter").length) {
        $(".js-filter li span").on("click", function (e) {
          e.preventDefault();
          $(this).parents("li").siblings().removeClass("is-active");
          $(this).parents("li").addClass("is-active");
          var filterName = $(this).parents("li").data("filter");
            $(".js-item-filter").hide();
          $(filterName).show();
            var textFilter = $(this).parents("li").data("text");
            $(".js-change-text").text(textFilter);
        });
    };
    
    if($(".js-filter-tag").length) {
        $(".js-filter-tag li a").on("click", function (e) {
          e.preventDefault();

          $('.line-follow').addClass('hided');

          if($(this).parent().hasClass('js-toggle-tags')) {
            $(this).parents('.list-tags').find('.hided').toggleClass('show');
            $(this).parents('.list-tags').find('.hide-tab').toggleClass('show');
            $(this).parents('.list-tags').find('.hide-mob').toggleClass('show');
            return;
          }

          if($(this).parent().data('filter') == '.all') {
            $('.line-follow').removeClass('hided');
          }

          $(this).parents("li").siblings().removeClass("active");
          $(this).parents("li").addClass("active");
          var filterName2 = $(this).parents("li").data("filter");
            $(".js-item-filter-tag").hide();
          $(filterName2).show();
        });
    };
    
    $(".js-anchor").on("click", function (e) {
      e.preventDefault();
      if($(this).attr("href") && $(this).attr("href") !== '#') {
        var anchorLink = $(this).attr("href");
      } else {
        var anchorLink = $(this).data("to");
      }
      var offset = $(anchorLink).offset().top;

      $("html, body").animate({
        scrollTop: offset
      }, 1000);
      
    });
    

    $('.js-load-more').on('click', function(e) {
      e.preventDefault();
      $(this).parents('.box-articles').find('.list-articles__item.hided').removeClass('hided');
      $(this).parent().remove();
    });
    
    $(".js-open-win").on("click", function (e) {
        e.preventDefault();
        var idWin = $(this).attr("href");
        $(idWin).slideDown();
        $('.window-overley').show();      
    });
    
    $(".js-close-win").on("click", function (e) {
        e.preventDefault();
        $('.window-review').slideUp();   
        $('.window-overley').hide();
        
    });
    $(".window-overley").on("click", function (e) {
        e.preventDefault();
        $('.js-close-win').click();           
    });
    
    if($('.styled').length) {
		$('.styled').styler({
            selectPlaceholder: "Job function"
        });
	};
    
    if ($('.js-select-number__title').length) {        
          $('.js-select-number__title').click(function() {
            var $dropdown = $(this).closest('.select-number').find('.select-number__drop');
            $dropdown.slideToggle(0);
          });

          $('.js-select-number__drop li').click(function() {
            var $selectedLi = $(this);
            var selectedValue = $selectedLi.find('span').text();

            var $title = $selectedLi.closest('.select-number').find('.select-number__title');
            $title.text(selectedValue);

            var $dropdown = $selectedLi.closest('.select-number__drop');
            $dropdown.slideUp(0);

            $selectedLi.addClass('active');
            $selectedLi.siblings().removeClass('active');

          });        
    }
    $('input[data-img-check]').change(function() {    
        var imageUrl = $(this).data('img-check');    
        $('.js-img-check').attr('src', imageUrl);
    });
    
    if ($('#js-form').length) {        
        $("#js-form").validate({
            rules: {             
              email: {
                required: true,
                email: true
              },
              phone: {
                required: true,        
              }
            },        
            messages: {                
                email: {
                  required: "Please complete this required field.",
                  email: "Enter correct E-mail"
                },
                phone: {
                    required: "Please complete this required field.",
                }     
            },
            invalidHandler: function(form, validator) {
                $('.js_success').addClass('hide-block');
                $('.js-btn-submit').removeClass('hide-block').addClass('err');
            },
            submitHandler: function (form) {
              $('.js_success').removeClass('hide-block');
              $('.js-btn-submit').addClass('hide-block');
            }
        });
    };

    if ($('#js-form-pos').length) {        
      $("#js-form-pos").validate({
          rules: {      
            name: {
              required: true,        
            },       
            email: {
              required: true,
              email: true
            },
            linkedin: {
              required: true,        
            }
          },        
          messages: {      
            name: {
                required: "Please complete this required field.",
              },          
              email: {
                required: "Please complete this required field.",
                email: "Enter correct E-mail"
              },
              linkedin: {
                  required: "Please complete this required field.",
              }     
          },
          invalidHandler: function(form, validator) {
              $('.js_success').addClass('hide-block');
              $('.js-btn-submit').removeClass('hide-block').addClass('err');
          },
          submitHandler: function (form) {
            $('.js_success').removeClass('hide-block');
            $('.js-btn-submit').addClass('hide-block');
          }
      });
  };
    
    if ($('.js-phone').length) { 
        $('.js-phone').intlTelInput({
          initialCountry: "us",
          separateDialCode: true,
          preferredCountries: ["fr", "us", "gb"],
          geoIpLookup: function(callback) {
            $.get('https://ipinfo.io', function() {}, "jsonp").always(function(resp) {
              var countryCode = (resp && resp.country) ? resp.country : "ua";
              callback(countryCode);
            });
          },
          utilsScript: "https://cdnjs.cloudflare.com/ajax/libs/intl-tel-input/11.0.14/js/utils.js"
        });

        //var mask1 = $('.js-phone').attr('placeholder').replace(/[0-9]/g, 0);


        //$('.js-phone').mask(mask1);

        $('.js-phone').on("countrychange", function(e, countryData) {
          //$('.js-phone').val('');
          //var mask1 = $('.js-phone').attr('placeholder').replace(/[0-9]/g, 0);
          //$('.js-phone').mask(mask1);
        });
    };
    
    $(".js-subscribe").on("click", function (e) {
        e.preventDefault();
        $('.win-subscribe, .win-subscribe-mask').addClass('active');           
    });
    $(".js-subscribe-close").on("click", function (e) {
        e.preventDefault();
        $('.win-subscribe, .win-subscribe-mask').removeClass('active');           
    });
    
    $('.leadership-list__cont').on('click', function(e) {
        e.preventDefault();
      $(this).parent().toggleClass('opened');
      $(this).parent().siblings().removeClass('opened');
    });
    
    if ($('.js-inf').length) { 
        $(window).scroll(function() {
          var scrollTop = $(window).scrollTop();
          var windowHeight = $(window).height();
          var infTop = $('.js-inf').offset().top;
          var infHeight = $('.js-inf').outerHeight();

          if (infTop < scrollTop + windowHeight / 2) {
            $('.js-inf').addClass('full');
            $('body').addClass('hide-video');
          } else {
            $('.js-inf').removeClass('full');
              $('body').removeClass('hide-video');
          }
        });
    };
    
    if($('.main-inf-about').length) {

      const chunkArray2 = (arr, size) => arr.length > size ? [arr.slice(0, size), ...chunkArray2(arr.slice(size), size)] : [arr];
      const imgsArr2 = chunkArray2($('.video-about-bg img'), 50);
        
      $(imgsArr2).each(function(i, item) {

        if (i == 0) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 2000);
        } else if (i == 1) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 3000);
        } else if (i == 2) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 4000);
        } else if (i == 3) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 5000);
        } else if (i == 4) {
          setTimeout(() => {
            $(item).each(function() {
              $(this).attr('src', $(this).data("src"));
              $(this).attr('srcset', $(this).data("srcset"));
            })
          }, 6000);
        }
      });


      $(window).scroll(function() {
        var scrollTop2 = $(this).scrollTop();

        var imgsBlockHeight2 = $('.js-inf-video').height();
        var imgsBlockTop2 = $('.js-inf-video').offset().top - imgsBlockHeight2 / 2;
    

        if (scrollTop2 >= imgsBlockTop2 && scrollTop2 < imgsBlockTop2 + imgsBlockHeight2) {
          let num3 = imgsBlockHeight2 / $('.video-about-bg  img').length;

				  let num2 = Math.round((scrollTop2 - imgsBlockTop2) / num3);

          if (num2 <= $('.video-about-bg  img').length - 1) {
            $('.video-about-bg  img').removeClass('active');
            $('.video-about-bg  img').eq(num2).addClass('active');
          }
        }
      });
    }
    
    if($('.js-play-video').length) {
        const playButtons = document.querySelectorAll('.play-video');
          
        playButtons.forEach((button) => {
            button.addEventListener('click', function() {             
              const parentVideoAbout = button.closest('.video-about');
              const videoPlayer = parentVideoAbout.querySelector('video');
              if (videoPlayer.paused) {
                videoPlayer.play();
                parentVideoAbout.classList.add('play')
              } else {                
                videoPlayer.pause();
                parentVideoAbout.classList.remove('play')
              }
            });
        });
    }
    
    
    $('.box-input-file input').change(function (e) {
      var fileName = e.target.files[0].name;
      $(this).parents('.box-input-file').addClass('active').find('.box-input-file__name').html(fileName);
    });
    
    
});

var handler = function(){
	
	var height_footer = $('footer').height();	
	var height_header = $('header').height();		
	
	
	
	var viewport_wid = viewport().width;
	var viewport_height = viewport().height;
	
	
	
}
$(window).bind('load', handler);
$(window).bind('resize', handler);



