object FormExport: TFormExport
  Left = 0
  Top = 0
  Caption = 'export to Tecplot360'
  ClientHeight = 299
  ClientWidth = 635
  Color = clMoneyGreen
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  PixelsPerInch = 96
  TextHeight = 13
  object CheckBox3D: TCheckBox
    Left = 24
    Top = 16
    Width = 233
    Height = 17
    Caption = 'export to Tecplot 360 3D xyz data .PLT'
    TabOrder = 0
  end
  object CheckBox2Dexport: TCheckBox
    Left = 24
    Top = 56
    Width = 233
    Height = 17
    Caption = 'export to Tecplot 360 2D xy data .PLT'
    TabOrder = 1
  end
  object CheckBox1Dexport: TCheckBox
    Left = 24
    Top = 96
    Width = 233
    Height = 17
    Caption = 'export to Tecplot 360 1D x data .PLT'
    TabOrder = 2
  end
  object Button1: TButton
    Left = 182
    Top = 240
    Width = 75
    Height = 25
    Caption = 'Apply'
    TabOrder = 3
    OnClick = Button1Click
  end
end
