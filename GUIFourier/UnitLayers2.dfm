object FormLayers: TFormLayers
  Left = 0
  Top = 0
  Caption = 'define Layers'
  ClientHeight = 502
  ClientWidth = 281
  Color = clMoneyGreen
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  OnClose = FormClose
  OnCreate = FormCreate
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 16
    Top = 16
    Width = 81
    Height = 13
    Caption = 'number of layers'
  end
  object Label3: TLabel
    Left = 24
    Top = 64
    Width = 147
    Height = 13
    Caption = '#  matherial  chickness, micron'
  end
  object Label4: TLabel
    Left = 95
    Top = 45
    Width = 18
    Height = 13
    Caption = 'Top'
  end
  object Label22: TLabel
    Left = 88
    Top = 448
    Width = 34
    Height = 13
    Caption = 'Bottom'
  end
  object ComboBox1: TComboBox
    Left = 112
    Top = 13
    Width = 33
    Height = 21
    ItemIndex = 0
    TabOrder = 0
    Text = '1'
    OnChange = ComboBox1Change
    Items.Strings = (
      '1'
      '2'
      '3'
      '4'
      '5'
      '6'
      '7'
      '8'
      '9')
  end
  object Panel1: TPanel
    Left = 8
    Top = 83
    Width = 257
    Height = 33
    TabOrder = 1
    Visible = False
    object Label2: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '9.'
    end
    object Label5: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox2: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit1: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel2: TPanel
    Left = 8
    Top = 122
    Width = 257
    Height = 33
    TabOrder = 2
    Visible = False
    object Label6: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '8.'
    end
    object Label7: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox3: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit2: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel3: TPanel
    Left = 8
    Top = 161
    Width = 257
    Height = 33
    TabOrder = 3
    Visible = False
    object Label8: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '7.'
    end
    object Label9: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox4: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit3: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel4: TPanel
    Left = 8
    Top = 200
    Width = 257
    Height = 33
    TabOrder = 4
    Visible = False
    object Label10: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '6.'
    end
    object Label11: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox5: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit4: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel5: TPanel
    Left = 8
    Top = 239
    Width = 257
    Height = 33
    TabOrder = 5
    Visible = False
    object Label12: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '5.'
    end
    object Label13: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox6: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit5: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel6: TPanel
    Left = 8
    Top = 278
    Width = 257
    Height = 33
    TabOrder = 6
    Visible = False
    object Label14: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '4.'
    end
    object Label15: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox7: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit6: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel7: TPanel
    Left = 8
    Top = 317
    Width = 257
    Height = 33
    TabOrder = 7
    Visible = False
    object Label16: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '3.'
    end
    object Label17: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox8: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit7: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel8: TPanel
    Left = 8
    Top = 356
    Width = 257
    Height = 33
    TabOrder = 8
    Visible = False
    object Label18: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '2.'
    end
    object Label19: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox9: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit8: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Panel9: TPanel
    Left = 8
    Top = 395
    Width = 257
    Height = 33
    TabOrder = 9
    object Label20: TLabel
      Left = 16
      Top = 8
      Width = 10
      Height = 13
      Caption = '1.'
    end
    object Label21: TLabel
      Left = 200
      Top = 16
      Width = 31
      Height = 13
      Caption = 'micron'
    end
    object ComboBox10: TComboBox
      Left = 32
      Top = 8
      Width = 73
      Height = 21
      ItemIndex = 0
      TabOrder = 0
      Text = 'GaN'
      Items.Strings = (
        'GaN'
        'Si'
        'SiC'
        'Sapphire'
        'diamond'
        'GaAs'
        'alumina'
        'AuSn'
        'Cu'
        'MD40'
        'eches'
        'gold'
        'Aluminium'
        'D16'
        'brass')
    end
    object Edit9: TEdit
      Left = 111
      Top = 8
      Width = 74
      Height = 21
      TabOrder = 1
    end
  end
  object Button1: TButton
    Left = 70
    Top = 467
    Width = 75
    Height = 25
    Caption = 'Return'
    TabOrder = 10
    OnClick = Button1Click
  end
end
