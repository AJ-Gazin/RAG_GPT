from gradio.themes.base import Base
from gradio.themes.utils import colors, sizes

class BusinessAnalyzerTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.indigo,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
        )
        self.set(
            # General theme settings
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_900",
            block_background_fill="white",
            block_background_fill_dark="*neutral_800",
            block_label_text_color="*neutral_700",
            block_label_text_color_dark="*neutral_200",
            block_title_text_weight="600",
            block_border_width="1px",
            block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",

            # Button styling
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_hover="*primary_700",
            button_primary_text_color="white",
            button_primary_background_fill_dark="*primary_700",
            button_primary_background_fill_hover_dark="*primary_600",
            button_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)",

            # Slider styling
            slider_color="*secondary_500",
            slider_color_dark="*secondary_400",

            # Card-like layout for answers
            block_border_color="*neutral_200",
            block_radius="*radius_lg",

            # Progress bar styling
            loader_color="*primary_500",
        )