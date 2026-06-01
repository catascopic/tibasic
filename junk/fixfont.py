from fontTools.ttLib import TTFont

font = TTFont('TI-83P-Font.ttf')

for table in font['cmap'].tables:
    a = table.cmap.get(0x0061)
    b = table.cmap.get(0x0062)
    table.cmap[0x0061] = b
    table.cmap[0x0062] = a

font.save('TI-83P-Font_mod.ttf')
