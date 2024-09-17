/* MOBILE HEADER TOGGLE */
$('body').on('click', '[data-header-toggle="true"]', function() {
    $('.header ul').slideToggle(100);
    $('body').toggleClass('overflow_hidden');
});

/* TRANSLATE WEATHER DESCRIPTION TO ICON CLASS AND DISPLAY TEXT */
function weatherDescriptionToClass(description) {
    var iconClass = 'full_clouds';
    var weatherDescription = description;
    switch (description) {
        case 'Partly Cloudy':
            iconClass = 'partly_cloudy';
            break;
        case 'Haze':
        case 'Overcast':
            iconClass = 'full_clouds';
            break;
        case 'Clear':
            iconClass = 'night';
            break;
        case 'Patchy Light Drizzle':
            iconClass = 'sun_rain_clouds';
            weatherDescription = 'Light Drizzle';
            break;
        case 'Sunny':
            iconClass = 'full_sun';
            break;
        case 'Patchy Rain Possible':
            iconClass = 'cloud_slight_rain';
            weatherDescription = 'Patchy Rain';
            break;
        case 'Light Rain':
        case 'Light Rain, Mist':
            iconClass = 'cloud_slight_rain';
            break;
        case 'Moderate Or Heavy Rain Shower':
            iconClass = 'rainy';
            weatherDescription = 'Heavy Rain';
            break;
        case 'Thunder':
            iconClass = 'thunder';
            break;
        default:
            iconClass = 'full_clouds';
            break;
            // some may be missing
    }
    return [iconClass, weatherDescription];
}
