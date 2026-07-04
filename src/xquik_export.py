XQUIK_TEXT_COLUMNS = ("text", "tweet", "tweet_text", "full_text", "content")
XQUIK_DATE_COLUMNS = ("created_at", "timestamp", "published_at", "date")
XQUIK_ID_COLUMNS = ("id", "tweet_id", "post_id")


def normalize_xquik_export_columns(df):
    """Map saved Xquik export columns into the dashboard upload schema."""
    renamed = df.copy()
    lower_to_original = {column.lower().strip(): column for column in renamed.columns}

    text_column = _first_present(lower_to_original, XQUIK_TEXT_COLUMNS)
    if text_column is not None and text_column != "text":
        renamed = renamed.rename(columns={text_column: "text"})

    date_column = _first_present(lower_to_original, XQUIK_DATE_COLUMNS)
    if date_column is not None and date_column != "date":
        renamed = renamed.rename(columns={date_column: "date"})

    id_column = _first_present(lower_to_original, XQUIK_ID_COLUMNS)
    if id_column is not None and id_column != "source_id":
        renamed = renamed.rename(columns={id_column: "source_id"})

    if text_column is not None and "platform" not in renamed.columns:
        renamed["platform"] = "Xquik"

    return renamed


def _first_present(field_map, candidates):
    for candidate in candidates:
        if candidate in field_map:
            return field_map[candidate]
    return None
