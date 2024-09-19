$(document).ready(function () {
    $('#load-articles').on('click', function (e) {
        e.preventDefault();
        var _this = $(this);
        if (_this.hasClass('prevent-click')) {
            return false;
        }

        _this.addClass('prevent-click');
        _this.fadeOut();
        if (currpage < maxnumpages) {
            currpage++;
            $.ajax({
                url: window.location.href + '?articles_page=' + currpage,
                type: "get",
                success: function (data) {
                    $('.list-articles:eq(0)').append($(data).find('.list-articles:eq(0)').html());
                    _this.removeClass('prevent-click');
                    if (currpage < maxnumpages) {
                        _this.fadeIn();
                    }
                }
            });
        }
    });

    if ($('#js-form2').length) {
        $("#js-form2").validate({
            rules: {
                email: {
                    required: true,
                    email: true
                },
                tel: {
                    required: true,
                },
                /*subscribe: {
                    required: true,
                }*/
            },
            messages: {
                email: {
                    required: "Please complete this required field.",
                    email: "Enter correct E-mail"
                },
                tel: {
                    required: "Please complete this required field.",
                },
                /*subscribe: {
                    required: "Please complete this required field.",
                }*/
            },
            invalidHandler: function(form, validator) {
                $('.js_success').addClass('hide-block');
                $('.js-btn-submit').removeClass('hide-block').addClass('err');
            },
            submitHandler: function (form) {
                $('.js-btn-submit').addClass('hide-block');
                let formData = new FormData(form);
                $.ajax({
                    url: '/wp-admin/admin-ajax.php',
                    data: formData,
                    type: 'POST',
                    dataType: 'json',
                    cache: false,
                    processData: false,
                    contentType: false,
                    beforeSend: function () {
                        $('.js-btn-submit').addClass('hide-block');
                    },
                    success: function (data) {
                        if (data.success) {
                            $('.js_success').removeClass('hide-block');
                            $('#js-form2 input[type="text"], #js-form2 input[type="tel"], #js-form2 input[type="email"], #js-form2 textarea').val('');
                            $('#js-form2 select[name="job"]').prop('selectedIndex',0).trigger( "change" );

                            setTimeout(function () {
                                $('.js_success').addClass('hide-block');
                                $('.js-btn-submit').removeClass('hide-block');
                            }, 5000)
                        } else {
                            $('.js-btn-submit').removeClass('hide-block');
                        }
                    }
                });
            }
        });
    }

    if ($('#js-form3').length) {
        $("#js-form3").validate({
            rules: {
                email: {
                    required: true,
                    email: true
                }
            },
            messages: {
                email: {
                    required: "Please complete this required field.",
                    email: "Enter correct E-mail"
                }
            },
            invalidHandler: function(form, validator) {
                //$('.js_success').addClass('hide-block');
                //$('.js-btn-submit').removeClass('hide-block').addClass('err');
            },
            submitHandler: function (form) {
                //$('.btn_subscribe').addClass('hide-block');
                let formData = new FormData(form);
                $.ajax({
                    url: '/wp-admin/admin-ajax.php',
                    data: formData,
                    type: 'POST',
                    dataType: 'json',
                    cache: false,
                    processData: false,
                    contentType: false,
                    beforeSend: function () {
                        $('.btn_subscribe').addClass('hide-block');
                    },
                    success: function (data) {
                        if (data.success) {
                            $('#js-form3 input[type="email"]').val('');
                            $('.btn_subscribe').removeClass('hide-block');
                            $('.line-btn-form__text').removeClass('hide-block');

                            setTimeout(function () {
                                $('.line-btn-form__text').addClass('hide-block');
                            }, 3000)
                        } else {
                            $('.btn_subscribe').removeClass('hide-block');
                        }
                    }
                });
            }
        });
    }

});