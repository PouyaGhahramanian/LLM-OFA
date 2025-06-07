import pandas as pd
import gc
from tqdm import tqdm

# ==========================
# File Paths
# ==========================
input_file = 'data_raw/articles.h5'
output_file = 'data_raw/articles_mapped.h5'

# ==========================
# Section Mapping
# ==========================
section_mapping = {
    # Business & Economy
    'Business Day': 'Business & Economy',
    'Real Estate': 'Business & Economy',
    'Your Money': 'Business & Economy',
    'The Upshot': 'Business & Economy',
    'Job Market': 'Business & Economy',

    # Politics & Government
    'U.S.': 'Politics & Government',
    'Washington': 'Politics & Government',
    'Opinion': 'Politics & Government',
    'Corrections': 'Politics & Government',
    'Week in Review': 'Politics & Government',

    # International News
    'World': 'International News',
    'New York': 'International News',
    'International Home': 'International News',
    'Global Home': 'International News',

    # Sports & Athletics
    'Sports': 'Sports & Athletics',

    # Health & Wellbeing
    'Health': 'Health & Wellbeing',
    'Well': 'Health & Wellbeing',
    'Climate': 'Health & Wellbeing',
    'Podcasts': 'Health & Wellbeing',
    'Neediest Cases': 'Health & Wellbeing',

    # Culture, Arts & Lifestyle
    'Arts': 'Culture, Arts & Lifestyle',
    'Movies': 'Culture, Arts & Lifestyle',
    'Theater': 'Culture, Arts & Lifestyle',
    'Books': 'Culture, Arts & Lifestyle',
    'Style': 'Culture, Arts & Lifestyle',
    'Fashion & Style': 'Culture, Arts & Lifestyle',
    'T Magazine': 'Culture, Arts & Lifestyle',
    'Magazine': 'Culture, Arts & Lifestyle',
    'Watching': 'Culture, Arts & Lifestyle',
    'Lens': 'Culture, Arts & Lifestyle',
    'Times Topics': 'Culture, Arts & Lifestyle',

    # Tech, Science & Education
    'Technology': 'Tech, Science & Education',
    'Science': 'Tech, Science & Education',
    'Multimedia/Photos': 'Tech, Science & Education',
    'Video': 'Tech, Science & Education',
    'Weather': 'Tech, Science & Education',
    'Times Insider': 'Tech, Science & Education',
    'Special Series': 'Tech, Science & Education',
    'Education': 'Tech, Science & Education',
    'The Learning Network': 'Tech, Science & Education',
    'Parenting': 'Tech, Science & Education',
    'College': 'Tech, Science & Education',
    'Guides': 'Tech, Science & Education',
    'Feeds': 'Tech, Science & Education',
    'Open': 'Tech, Science & Education',

    # Lifestyle & Leisure
    'Travel': 'Lifestyle & Leisure',
    'Food': 'Lifestyle & Leisure',
    'Home & Garden': 'Lifestyle & Leisure',
    'Great Homes & Destinations': 'Lifestyle & Leisure',
    'UrbanEye': 'Lifestyle & Leisure',
    'Smarter Living': 'Lifestyle & Leisure',
    'At Home': 'Lifestyle & Leisure',
    'Today’s Paper': 'Lifestyle & Leisure',
    'NYT Now': 'Lifestyle & Leisure',
    'The Weekly': 'Lifestyle & Leisure',
    'The New York Times Presents': 'Lifestyle & Leisure',
    'Polls': 'Lifestyle & Leisure',
    'Obituaries': 'Lifestyle & Leisure',
    'Book Review': 'Lifestyle & Leisure',
    'Blogs': 'Lifestyle & Leisure',
    'Archives': 'Lifestyle & Leisure',
    'Briefing': 'Lifestyle & Leisure',
    'Slideshows': 'Lifestyle & Leisure',
    'Critic\'s Choice': 'Lifestyle & Leisure',
    'Reader Center': 'Lifestyle & Leisure'
}

# ==========================
# HDF5 Storage Tuning
# ==========================
min_itemsize = {
    'web_url': 300,
    'text': 20000,
    'section_name': 100
}

# ==========================
# Read, Map, Save
# ==========================
def main():
    print("🔍 Reading original HDF5 file...")
    df = pd.read_hdf(input_file, key='data')
    print(f"📰 Total articles read: {len(df):,}")

    print("🗂️ Mapping section names...")
    df['mapped_section'] = df['section_name'].map(section_mapping).fillna('Other')

    print("💾 Saving mapped data to new HDF5 file...")
    df.to_hdf(
        output_file,
        key='data',
        mode='w',
        format='table',
        complevel=5,
        complib='blosc',
        min_itemsize=min_itemsize
    )

    print(f"✅ Done! Mapped file saved at: {output_file}")


if __name__ == '__main__':
    main()