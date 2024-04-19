from woodnet.inference.utils import parse_checkpoint_fname

class Test_parse_checkpoint_fname:

    def test_with_basic_valid_qualified_fname(self):
        prefix = 'chkpt'
        qualifier = 'optimal'
        UUID = '1701-phaser-ferengi-wolf-359'
        suffix = '.pth'
        fname = '_'.join((prefix, qualifier, UUID)) + suffix

        parts = parse_checkpoint_fname(fname)
        assert parts.prefix == prefix
        assert parts.qualifier == qualifier
        assert parts.UUID == UUID


    def test_with_basic_valid_unqualified_fname(self):
        prefix = 'chkpt'
        qualifier = None
        UUID = '1701-phaser-ferengi-wolf-359'
        suffix = '.pth'
        fname = '_'.join((prefix, UUID)) + suffix

        parts = parse_checkpoint_fname(fname)
        assert parts.prefix == prefix
        assert parts.qualifier == qualifier
        assert parts.UUID == UUID