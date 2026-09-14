import pandas as pd
import pydeck as pdk

coords = pd.read_csv('sector_coordinates.csv')
price_df = pd.read_csv('gurgaon_properties_post_feature_selection_v2.csv')
sector_avg = price_df.groupby('sector')['price'].mean().round(2).reset_index()
merged = sector_avg.merge(coords, on='sector', how='inner')
merged['price_str'] = merged['price'].apply(lambda p: f'₹ {p:.2f} Cr')
merged['radius'] = merged['price'].apply(lambda p: max(200, min(1400, int(p * 180))))

min_p, max_p = merged['price'].min(), merged['price'].max()
def get_color(p):
    ratio = (p - min_p) / (max_p - min_p) if max_p > min_p else 0.5
    r = int(0 + ratio * 200)
    g = int(120 - ratio * 40)
    b = int(219 - ratio * 100)
    return [r, g, b, 200]

merged['color'] = merged['price'].apply(get_color)

view_state = pdk.ViewState(
    latitude=float(merged['latitude'].mean()),
    longitude=float(merged['longitude'].mean()),
    zoom=10.2,
    pitch=0,
)

layer = pdk.Layer(
    'ScatterplotLayer',
    data=merged,
    get_position=['longitude', 'latitude'],
    get_color='color',
    get_radius='radius',
    pickable=True,
    auto_highlight=True,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={'html': '<b>Sector:</b> {sector}<br/><b>Average Price:</b> {price_str}', 'style': {'backgroundColor': '#041533', 'color': 'white'}},
    map_style=None
)
print('Pydeck test passed successfully! Total sectors mapped:', len(merged))
