// team choser
function setSwitcherPos(el) {
  setTimeout(() => {
    const switchWidth = $(el).outerWidth();
    const switchPos = $(el).position().left;
    $(el)
      .parents(".teams__choser-switcher")
      .find(".teams__choser-decor")
      .width(switchWidth);
    $(el)
      .parents(".teams__choser-switcher")
      .find(".teams__choser-decor")
      .css("left", switchPos);
  }, 100);
}

var handler = function () {
  setTimeout(() => {
    setSwitcherPos(".teams__choser-item.active");
  }, 1000);
};
$(window).bind("resize", handler);
$(window).bind("load", handler);

$(".teams__choser-item").on("click", function () {
  const thisSwitch = $(this);
  setSwitcherPos(thisSwitch);
  $(".teams__choser-item").removeClass("active");
  $(this).addClass("active");

  $(".teams__box").attr("data-team", thisSwitch.data("team"));

  $("[data-team-section]").removeClass("active");
  $(`[data-team-section='${thisSwitch.data("team")}']`).addClass("active");

  ScrollTrigger.refresh();

  count = 0;
  
  if (thisSwitch.data("team") == "team1") {
    $("footer").addClass("black-footer");
    //$(".teams__steps").attr("data-color", "black");
    //$(".teams__principles").attr("data-color", "black");
    //$(".teams__imgs").attr("data-color", "black");
    //$(".teams__benefits").attr("data-color", "black");
  } else {
    $("footer").removeClass("black-footer");
    //$(".teams__steps").attr("data-color", "white");
    //$(".teams__principles").attr("data-color", "white");
    //$(".teams__imgs").attr("data-color", "white");
    //$(".teams__benefits").attr("data-color", "white");
  }

  refreshTeamSliders();
  refreshTestSliders();
});


// step change

function changeStep(index) {
  $("[data-team-section].active .teams__steps-item-title")
    .parent()
    .removeClass("active");
  $("[data-team-section].active .teams__steps-nav-item-title")
    .parent()
    .removeClass("active");
  $("[data-team-section].active .teams__steps-item-title")
    .parent()
    .eq(index)
    .addClass("active");
  $("[data-team-section].active .teams__steps-nav-item-title")
    .parent()
    .eq(index)
    .addClass("active");

  $(".teams__steps-nav-line").css(
    "height",
    (parseInt(index + 1) /
      $("[data-team-section].active .teams__steps-nav-item").length) *
      100 +
      "%"
  );
}
let count = 0;
function changeStepIndex() {
  changeStep(count);
  count++;
  if (count >= $("[data-team-section].active .teams__steps-nav-item").length) {
    count = 0;
  }
}
changeStepIndex();
const stepsAutoplay = setInterval(changeStepIndex, 5000);

$(".teams__steps-item-title, .teams__steps-nav-item-title").on(
  "hover",
  function () {
    count = $(this).parent().index();
    changeStep(count);
  }
);

// teams images slider
if ($(".js-teams__imgs").length) {
  var slidesArr2 = [];
  $(".js-teams__imgs").each(function( index ) {
    var splide = new Splide(this, {
      autoWidth: true,
      focus: 0,
      omitEnd: true,
      arrows: false,
      pagination: false,
      gap: "1.2rem",
      breakpoints: {
        1024: {
          gap: "1.7rem",
        },
        767: {
          autoWidth: false,
          perPage: 1,
        },
      },
    });
    splide.on( 'mounted', function () {
      var currentSlide = splide.index + 1;
      $('.slider-num-current').text(currentSlide);
      $('.slider-num-total').text(splide.Components.Slides.getLength());
    });
    splide.mount();
    splide.on('move', function (newIndex, prevIndex, destIndex) {
      var currentSlide = splide.index + 1;
      $('.slider-num-current').text(currentSlide);
    });
    splide.on( 'refresh', function () {
      var currentSlide = splide.index + 1;
      $('.slider-num-current').text(currentSlide);
      $('.slider-num-total').text(splide.Components.Slides.getLength());
    });

    slidesArr2.push(splide)
  });

  function refreshTeamSliders() {
    slidesArr2.forEach((item) => {
      item.refresh();
    })
  }
}

// teams test

if ($(".teams__test").length) {
  var slidesArr3 = [];
  $('.teams__test').each(function(i) {
    const sliderQuestions = $(this).find('.js-teams__test-questions')[0];
    const sliderPercent = $(this).find('.js-teams__test-res-percent')[0];
    const test = $(this);

    var splideTest = new Splide(sliderQuestions, {
      type: "slide",
      direction: "ttb",
      heightRatio: 0.7,
      arrows: false,
      pagination: false,
      loop: false,
      drag: false,
      height: 400
    });
    splideTest.mount();
    splideTest.on("moved", (nextIndex) => {
      $(this).find(".teams__test-nav").find("span").removeClass("active");
      $(this).find(".teams__test-nav").find("span").eq(nextIndex).addClass("active");
    });

    var splidePercent = new Splide(sliderPercent, {
      type: "slide",
      direction: "ttb",
      heightRatio: 0.7,
      arrows: false,
      pagination: false,
      loop: false,
      drag: false,
      perPage: 1,
      height: 400
    });
    splidePercent.mount();

    let yesCount = 0;

    $(test).find(".teams__test-action-item").on("click", function (e) {
      e.preventDefault();
      splideTest.go(">");
      if (splideTest.Components.Controller.getNext() === -1) {
        $(test).addClass("active");
  
        setTimeout(() => {
          $(test).find(".teams__test-res-match").addClass("active");
        }, 200);
  
        setTimeout(() => {
          $(test).find('.teams__test-res-again span').text($(test).find('.teams__test-res-percent .splide__slide.is-active').text().trim());
  
          if($(test).find('.teams__test-res-percent .splide__slide.is-active').text().trim().slice(0, -1) >= 50.1) {
            const textFin = $(test).find('.teams__test-res-match-text').data('text1');
            $(test).find('.teams__test-res-match-text').text(textFin);
              console.log(textFin)
          } else {
            const textFin = $(test).find('.teams__test-res-match-text').data('text2');
            $(test).find('.teams__test-res-match-text').text(textFin);
              console.log(textFin)
          }
        }, 500);
  
      }
  
      if ($(this).text().trim() == "No") {
        if (splideTest.index == 1) {
          $(test).find(".teams__test-res-text").text("");
        }
      } else {
        $(test).find(".teams__test-res-text").text("Match");
        yesCount++;
        splidePercent.go(">");
        const resPercent = (yesCount * 16.7).toFixed() + "%";
        // $(".teams__test-res-percent").text(resPercent);
  
        if ($(window).width() <= 1024) {
          $(test).find(".teams__test-res-scale").css("width", resPercent);
        } else {
          $(test).find(".teams__test-res-scale").css("height", resPercent);
        }
      }
    });
  
    $(this).find(".teams__test-res-again").on("click", function (e) {
      e.preventDefault();
      $(test).removeClass("active");
  
      if ($(window).width() < 1024) {
        $(test).find(".teams__test-res-scale").css("width", 0);
      } else {
        $(test).find(".teams__test-res-scale").css("height", 0);
      }
  
      $(test).find(".teams__test-res-text").text(
        "Let’s see how your principles align with the company’s"
      );
      $(test).find(".teams__test-res-match").removeClass("active");
      yesCount = 0;
      splideTest.go(0);
      splidePercent.go(0);
    });

    slidesArr3.push(splideTest, splidePercent)
  });

  function refreshTestSliders() {
    slidesArr3.forEach((item) => {
      item.refresh();
    })
  }
  refreshTestSliders();
}

// benefits animation

gsap.registerPlugin(ScrollTrigger);
let mm = gsap.matchMedia();
ScrollTrigger.refresh();

if($('.teams__benefits-items').length) {
  $(".teams__benefits-items").each(function( index ) {
    const benefitsItemsBox = this;
    const benefitsItems = gsap.utils.toArray($(this).find('.teams__benefits-item'));
    
    mm.add("(min-width: 1025px)", () => {
      gsap
        .timeline({
          scrollTrigger: {
            trigger: benefitsItemsBox,
            end: "top -=200",
            scrub: true,
            // pin: true,
          },
        })
        .from(benefitsItems, {
          opacity: 0,
          y: 200,
          stagger: 0.1,
          ease: "power2.inOut",
        });
    });
    
    mm.add("(max-width: 1024px)", () => {
      gsap
        .timeline({
          scrollTrigger: {
            trigger: benefitsItemsBox,
            end: "bottom bottom-=300",
            scrub: true,
          },
        })
        .from(benefitsItems, {
          opacity: 0,
          y: 100,
          stagger: 0.1,
          ease: "power2.inOut",
          duration: 1,
        });
    });
  });
}

// feedback slider

if ($(".js-slider-logo").length) {
  $(".js-slider-logo").each(function( index ) {
    const sliderComments = $(this).find('.js-slider-comments')[0];
    const sliderLogos = $(this).find('.js-logo-slider')[0];

    var commentsSplide = new Splide(sliderComments, {
      type: "fade",
      rewind: true,
      autoplay: true,
      pagination: false,
      arrows: false,
      cover: true,
      autoplay: true,
      interval: 5000,
    });
  
    var navSplide = new Splide(sliderLogos, {
      rewind: true,
      isNavigation: true,
      arrows: false,
      pagination: false,
      perPage: 5,
      drag: false,
      autoplay: true,
      interval: 5000,
      breakpoints: {
        1024: {
          perPage: 4,
        },
        768: {
          perPage: 1,
        },
      },
    });
  
    commentsSplide.sync(navSplide);
    commentsSplide.mount();
    navSplide.mount();

  });
}
