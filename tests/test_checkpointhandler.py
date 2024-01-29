from woodnet.checkpointhandler import generate_filename



def test_prefix():
    startstring = 'picard'
    filename = generate_filename(prefix=startstring)
    assert filename.startswith(startstring)


def test_qualifier():
    qualifier = 'superduper'
    filename = generate_filename(qualifier=qualifier)
    assert f'-{qualifier}-' in filename