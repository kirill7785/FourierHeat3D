object FormTopology: TFormTopology
  Left = 0
  Top = 0
  Caption = #1047#1072#1093#1072#1088#1086#1074', '#1040#1089#1074#1072#1076#1091#1088#1086#1074#1072' remake 2021'
  ClientHeight = 305
  ClientWidth = 635
  Color = clMoneyGreen
  CustomTitleBar.CaptionAlignment = taCenter
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  Menu = MainMenu1
  OldCreateOrder = False
  OnCreate = FormCreate
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 17
    Top = 284
    Width = 90
    Height = 13
    Caption = 'Thermal resistance'
  end
  object LabelThermalresistance: TLabel
    Left = 113
    Top = 284
    Width = 3
    Height = 13
  end
  object Label3: TLabel
    Left = 168
    Top = 284
    Width = 21
    Height = 13
    Caption = 'C/W'
  end
  object Labelwait: TLabel
    Left = 17
    Top = 265
    Width = 78
    Height = 13
    Caption = '                          '
  end
  object Label2: TLabel
    Left = 8
    Top = 8
    Width = 117
    Height = 13
    Caption = 'dimension of sources (x)'
  end
  object Label4: TLabel
    Left = 304
    Top = 11
    Width = 31
    Height = 13
    Caption = 'micron'
  end
  object Label5: TLabel
    Left = 8
    Top = 65
    Width = 142
    Height = 13
    Caption = 'distance between sources (x)'
  end
  object Label6: TLabel
    Left = 304
    Top = 65
    Width = 31
    Height = 13
    Caption = 'micron'
  end
  object Label7: TLabel
    Left = 8
    Top = 38
    Width = 131
    Height = 13
    Caption = 'source number in a x group'
  end
  object Label8: TLabel
    Left = 8
    Top = 150
    Width = 120
    Height = 13
    Caption = 'dimension of sources (y) '
  end
  object Label9: TLabel
    Left = 304
    Top = 150
    Width = 31
    Height = 13
    Caption = 'micron'
  end
  object Label10: TLabel
    Left = 8
    Top = 123
    Width = 137
    Height = 13
    Caption = 'gap size between groups (x)'
  end
  object Label11: TLabel
    Left = 304
    Top = 123
    Width = 31
    Height = 13
    Caption = 'micron'
  end
  object Label12: TLabel
    Left = 17
    Top = 93
    Width = 102
    Height = 13
    Caption = 'number of groups (x)'
  end
  object Label13: TLabel
    Left = 8
    Top = 184
    Width = 131
    Height = 13
    Caption = 'source number in a y group'
  end
  object Label14: TLabel
    Left = 8
    Top = 215
    Width = 142
    Height = 13
    Caption = 'distance between sources (y)'
  end
  object Label15: TLabel
    Left = 304
    Top = 211
    Width = 31
    Height = 13
    Caption = 'micron'
  end
  object Button1: TButton
    Left = 20
    Top = 234
    Width = 75
    Height = 25
    Caption = 'Run'
    TabOrder = 0
    OnClick = Button1Click
  end
  object Edit1: TEdit
    Left = 168
    Top = 8
    Width = 121
    Height = 21
    TabOrder = 1
  end
  object Edit2: TEdit
    Left = 168
    Top = 62
    Width = 121
    Height = 21
    TabOrder = 2
    Text = '0'
  end
  object ComboBox1: TComboBox
    Left = 168
    Top = 35
    Width = 121
    Height = 21
    ItemIndex = 0
    TabOrder = 3
    Text = '1'
    Items.Strings = (
      '1'
      '2'
      '3'
      '4'
      '5'
      '6'
      '7'
      '8'
      '9'
      '10'
      '11'
      '12'
      '13'
      '14'
      '15'
      '16'
      '17'
      '18'
      '19'
      '20'
      '21'
      '22'
      '23'
      '24'
      '25'
      '26'
      '27'
      '28'
      '29'
      '30'
      '31'
      '32'
      '33'
      '34'
      '35'
      '36'
      '37'
      '38'
      '39'
      '40'
      '41'
      '42'
      '43'
      '44'
      '45'
      '46'
      '47'
      '48'
      '49'
      '50'
      '51'
      '52'
      '53'
      '54'
      '55'
      '56'
      '57'
      '58'
      '59'
      '60'
      '61'
      '62'
      '63'
      '64'
      '65'
      '66'
      '67'
      '68'
      '69'
      '70'
      '71'
      '72'
      '73'
      '74'
      '75'
      '76'
      '77'
      '78'
      '79'
      '80'
      '81'
      '82'
      '83'
      '84'
      '85'
      '86'
      '87'
      '88'
      '89'
      '90'
      '91'
      '92'
      '93'
      '94'
      '95'
      '96'
      '97'
      '98'
      '99'
      '100'
      '101'
      '102'
      '103'
      '104'
      '105'
      '106'
      '107'
      '108'
      '109'
      '110'
      '111'
      '112'
      '113'
      '114'
      '115'
      '116'
      '117'
      '118'
      '119'
      '120'
      '121'
      '122'
      '123'
      '124'
      '125'
      '126'
      '127'
      '128'
      '129'
      '130'
      '131'
      '132'
      '133'
      '134'
      '135'
      '136'
      '137'
      '138'
      '139'
      '140'
      '141'
      '142'
      '143'
      '144'
      '145'
      '146'
      '147'
      '148'
      '149'
      '150'
      '151'
      '152'
      '153'
      '154'
      '155'
      '156'
      '157'
      '158'
      '159'
      '160'
      '161'
      '162'
      '163'
      '164'
      '165'
      '166'
      '167'
      '168'
      '169'
      '170'
      '171'
      '172'
      '173'
      '174'
      '175'
      '176'
      '177'
      '178'
      '179'
      '180')
  end
  object Edit3: TEdit
    Left = 168
    Top = 147
    Width = 121
    Height = 21
    TabOrder = 4
  end
  object Edit4: TEdit
    Left = 168
    Top = 120
    Width = 121
    Height = 21
    TabOrder = 5
    Text = '0'
  end
  object ComboBox2: TComboBox
    Left = 168
    Top = 93
    Width = 121
    Height = 21
    AutoDropDown = True
    ItemIndex = 0
    TabOrder = 6
    Text = '1'
    Items.Strings = (
      '1'
      '2'
      '3'
      '4'
      '5'
      '6')
  end
  object Button2: TButton
    Left = 128
    Top = 234
    Width = 75
    Height = 25
    Caption = 'Layers'
    TabOrder = 7
    OnClick = Button2Click
  end
  object ComboBox3: TComboBox
    Left = 168
    Top = 181
    Width = 121
    Height = 21
    ItemIndex = 0
    TabOrder = 8
    Text = '1'
    Items.Strings = (
      '1'
      '2'
      '3'
      '4'
      '5'
      '6'
      '7'
      '8'
      '9'
      '10'
      '11'
      '12'
      '13'
      '14'
      '15'
      '16'
      '17'
      '18'
      '19'
      '20'
      '21'
      '22'
      '23'
      '24'
      '25'
      '26'
      '27'
      '28'
      '29'
      '30'
      '31'
      '32'
      '33'
      '34'
      '35'
      '36'
      '37'
      '38'
      '39'
      '40'
      '41'
      '42'
      '43'
      '44'
      '45'
      '46'
      '47'
      '48'
      '49'
      '50'
      '51'
      '52'
      '53'
      '54'
      '55'
      '56'
      '57'
      '58'
      '59'
      '60'
      '61'
      '62'
      '63'
      '64'
      '65'
      '66'
      '67'
      '68'
      '69'
      '70'
      '71'
      '72'
      '73'
      '74'
      '75'
      '76'
      '77'
      '78'
      '79'
      '80')
  end
  object Edit5: TEdit
    Left = 168
    Top = 208
    Width = 121
    Height = 21
    TabOrder = 9
    Text = '0'
  end
  object MainMenu1: TMainMenu
    Left = 352
    Top = 32
    object File1: TMenuItem
      Caption = 'File'
      object Close1: TMenuItem
        Caption = 'Close'
        OnClick = Close1Click
      end
    end
    object File2: TMenuItem
      Caption = 'Mesh'
    end
    object Define1: TMenuItem
      Caption = 'Define'
      object Define2: TMenuItem
        Caption = 'Conditions'
        OnClick = Define2Click
      end
      object Export1: TMenuItem
        Caption = 'Export'
        OnClick = Export1Click
      end
      object Matherials1: TMenuItem
        Caption = 'Matherials'
        OnClick = Matherials1Click
      end
      object Launcher1: TMenuItem
        Caption = 'Launcher'
        OnClick = Launcher1Click
      end
    end
  end
end
