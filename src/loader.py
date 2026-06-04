import pandas as pd


def load_excel(file):
    """
    Load seluruh sheet yang dibutuhkan dari file SLA DSE.

    Parameters
    ----------
    file : str atau UploadedFile
        Path file Excel atau file upload Streamlit

    Returns
    -------
    dict
        Dictionary berisi dataframe:
        - df_im3
        - df_3id
        - df_im3_ec
        - df_3id_ec
    """

    df_im3 = pd.read_excel(
        file,
        sheet_name='DSE_IM3',
        header=2
    )

    df_3id = pd.read_excel(
        file,
        sheet_name='DSE_3ID',
        header=2
    )

    df_im3_ec = pd.read_excel(
        file,
        sheet_name='DSE_IM3_EC',
        header=3
    )

    df_3id_ec = pd.read_excel(
        file,
        sheet_name='DSE_3ID_EC',
        header=3
    )

    # Tambah source brand
    df_im3['BRAND_SRC'] = 'IM3'
    df_3id['BRAND_SRC'] = '3ID'

    df_im3_ec['BRAND_SRC'] = 'IM3'
    df_3id_ec['BRAND_SRC'] = '3ID'

    return {
        'df_im3': df_im3,
        'df_3id': df_3id,
        'df_im3_ec': df_im3_ec,
        'df_3id_ec': df_3id_ec
    }